import torch
import argparse
import triton.language as tl
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        moe_align_block_size,
        try_get_optimal_moe_config,
        invoke_fused_moe_kernel,
        moe_sum_reduce_torch_compile,
        )
from sgl_kernel import silu_and_mul
import alpha_moe

from sglang.srt.server_args import set_global_server_args_for_scheduler

# Sglang fused moe requires this set
class FakeServerArgs:
    enable_deterministic_inference=False

set_global_server_args_for_scheduler(FakeServerArgs())

def interleave_tensor(tensor, rep=8):
    M, N, K = tensor.shape

    first_half = tensor[:, :(N//2), :]
    second_half = tensor[:, (N//2):, :]

    first_chunks = first_half.view(M, (N//(2*rep)), rep, K)
    second_chunks = second_half.view(M, (N//(2*rep)), rep, K)

    interleaved = torch.stack([first_chunks, second_chunks], dim=2)
    result = interleaved.view(M, N, K)

    return result.contiguous()


def generate_topk_ids(num_experts, num_tokens, top_k, balancedness=1.0):
    uniform = torch.ones(num_experts) / num_experts
    skewed = torch.zeros(num_experts); skewed[0] = 1.0
    probs = balancedness * uniform + (1 - balancedness) * skewed

    topk_ids = torch.multinomial(probs, num_tokens * top_k, replacement=True)
    topk_ids = topk_ids.view(num_tokens, top_k)
    return topk_ids


def test_configuration(num_tokens, E, N, K, top_k, block_m, bn, wn, stages,
                       atol=1e-2, rtol=1e-1):
    torch.manual_seed(42)
    block_shape = [128, 128]
    routed_scaling_factor=2.5

    w1 = torch.randn((E, N, K)) * 50
    w2 = torch.randn((E, K, N//2)) * 50

    w1 = w1.to(torch.float8_e4m3fn).to("cuda:0")
    w2 = w2.to(torch.float8_e4m3fn).to("cuda:0")

    w1_scale = torch.empty((E, w1.shape[1]//block_shape[0], w1.shape[2]//block_shape[1]), dtype=torch.float32).normal_(mean=0, std=0.001)
    w2_scale = torch.empty((E, w2.shape[1]//block_shape[0], w2.shape[2]//block_shape[1]), dtype=torch.float32).normal_(mean=0, std=0.001)

    topk_ids = generate_topk_ids(E-1, num_tokens, top_k-1)
    topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens).view(num_tokens,1).to(torch.int32)*(E-1)))
    topk_weights = torch.nn.functional.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32), dim=-1)

    x = torch.empty((num_tokens, K), dtype=torch.bfloat16).normal_(mean=0, std=0.05)
    x_q, x_scale = sglang_per_token_group_quant_fp8(x, block_shape[1])

    config_dtype = 'fp8_w8a8'
    triton_config = try_get_optimal_moe_config(w1.shape, w2.shape, top_k, config_dtype, block_shape=block_shape, M=num_tokens)
    compute_type = tl.bfloat16

    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(
        topk_ids, triton_config["BLOCK_SIZE_M"], E
    )

    out_triton_up = torch.empty((num_tokens, top_k, w1.shape[1]), device=x.device, dtype=x.dtype)
    out_triton_swiglu = torch.empty((num_tokens*top_k, w1.shape[1]//2), device=x.device, dtype=x.dtype)
    out_triton_down = torch.empty((num_tokens, top_k, x.shape[1]), device=x.device, dtype=x.dtype)
    out_sglang = torch.empty_like(x)

    invoke_fused_moe_kernel(x, w1, None, out_triton_up, None, w1_scale, None, topk_weights, topk_ids,
                            sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton,
                            False, top_k, triton_config, compute_type, True, False, False, False, False, block_shape)
    silu_and_mul(out_triton_up.view(-1, w1.shape[1]), out_triton_swiglu)
    invoke_fused_moe_kernel(out_triton_swiglu, w2, None, out_triton_down, None, w2_scale, None, topk_weights, topk_ids,
                            sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton,
                            True, 1, triton_config, compute_type, True, False, False, False, False, block_shape)
    moe_sum_reduce_torch_compile(out_triton_down.view(*out_triton_down.shape), out_sglang, routed_scaling_factor)

    w1_swiglu = interleave_tensor(w1)
    w1_scale_swiglu = interleave_tensor(w1_scale, rep=1)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_m, E)
    out = torch.zeros_like(x)

    torch.ops.alpha_moe.fused_moe_w8a8_up_down(
        x_q, x_scale, w1_swiglu, w1_scale_swiglu, w2, w2_scale,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_weights, out, top_k,
        block_m, bn, wn, stages, routed_scaling_factor
    )
    # print(out[0, :10])
    # print(out_sglang[0, :10])

    diff = torch.abs(out - out_sglang)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    passed = torch.allclose(out, out_sglang, atol=atol, rtol=rtol)

    return passed, mean_diff, max_diff


def parse_arguments():
    parser = argparse.ArgumentParser(description='MOE Correctness Test')

    parser.add_argument('--batch-sizes',
                        type=int,
                        nargs='+',
                        default=[8, 32, 128, 256],
                        help='Batch sizes to test')

    parser.add_argument('--N-values',
                        type=int,
                        nargs='+',
                        default=[256, 512],
                        help='N dimension values')

    parser.add_argument('--K-values',
                        type=int,
                        nargs='+',
                        default=[7168, 4096],
                        help='K dimension values')

    parser.add_argument('--E-values',
                        type=int,
                        nargs='+',
                        default=[257, 8, 37],
                        help='num_experts values')

    parser.add_argument('--top-k',
                        type=int,
                        default=9,
                        help='top-k experts')

    parser.add_argument('--atol',
                        type=float,
                        default=2e-2,
                        help='Absolute tolerance')

    parser.add_argument('--rtol',
                        type=float,
                        default=1e-1,
                        help='Relative tolerance')

    return parser.parse_args()


torch.set_default_device("cuda:0")


if __name__ == "__main__":
    args = parse_arguments()

    batch_sizes = args.batch_sizes
    N_values = args.N_values
    K_values = args.K_values
    E_values = args.E_values
    top_k = args.top_k

    configs_to_test = []
    for block_m in range(8, 129, 8):
        for bn, wn in [(64, 4), (32, 8)]:
            for stages in range(1, 6):
                # Goes out of shared memory
                if stages == 5 and block_m > 100:
                    continue
                configs_to_test.append((block_m, bn, wn, stages))

    total_tests = 0
    passed_tests = 0

    for E in E_values:
        for N in N_values:
            for K in K_values:
                for num_tokens in batch_sizes:
                    for block_m, bn, wn, stages in configs_to_test:
                        if num_tokens < block_m and block_m != 16:
                            continue
                        total_tests += 1
                        try:
                            passed, mean_diff, max_diff = test_configuration(
                                num_tokens, E, N, K, top_k,
                                block_m, bn, wn, stages,
                                atol=args.atol,
                                rtol=args.rtol
                            )

                            if passed:
                                passed_tests += 1
                            else:
                                print(f"FAIL: E={E}, N={N}, K={K}, tokens={num_tokens}, "
                                      f"block_m={block_m}, bn={bn}, wn={wn}, stages={stages} "
                                      f"(mean_diff={mean_diff:.2e}, max_diff={max_diff:.2e})")
                        except Exception as e:
                            print(f"ERROR: E={E}, N={N}, K={K}, tokens={num_tokens}, "
                                  f"block_m={block_m}, bn={bn}, wn={wn}, stages={stages}: {e}")

    print(f"\n{'='*80}")
    print(f"Summary: {passed_tests}/{total_tests} tests passed")
