import torch
import json
import argparse
import triton.language as tl
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    moe_align_block_size,
    try_get_optimal_moe_config,
    invoke_fused_moe_kernel,
    moe_sum_reduce_triton,
    moe_sum_reduce_torch_compile,
)
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sgl_kernel import silu_and_mul
from statistics import mean
import alpha_moe
from alpha_moe_python.utils import get_best_config

# Sglang fused moe requires this set
class FakeServerArgs:
    enable_deterministic_inference=False

set_global_server_args_for_scheduler(FakeServerArgs())

def bench_kineto(fn, kernel_name: str = "moe", num_tests: int = 25):
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
        for i in range(2):
            for _ in range(num_tests):
                x = torch.randn((8, 8192, 8192), dtype=torch.float, device='cuda')
                fn()
                torch.cuda.synchronize()
            prof.step()

    times = []
    for e in prof.profiler.function_events:
        if kernel_name in e.name:
            times.append(e.device_time_total)
    return sum(times)/len(times)


def generate_topk_ids(num_experts, num_tokens, top_k, balancedness=1.0):
    uniform = torch.ones(num_experts) / num_experts
    skewed = torch.zeros(num_experts); skewed[0] = 1.0
    probs = balancedness * uniform + (1 - balancedness) * skewed

    topk_ids = torch.multinomial(probs, num_tokens * top_k, replacement=True)
    topk_ids = topk_ids.view(num_tokens, top_k)
    return topk_ids


def get_stats(activated_experts, num_tokens, top_k, w1, w2, w1_scale, w2_scale, hidden_size, block_shape):
    flops_1 = 2 * num_tokens * w1.shape[1] * w1.shape[2]
    flops_2 = 2 * num_tokens * top_k * w2.shape[1] * w2.shape[2]

    mem_1 = (activated_experts * w1.shape[1] * w1.shape[2] * w1.element_size() +
             activated_experts * w1.shape[1]//block_shape[0] * w1.shape[2]//block_shape[1] * w1_scale.element_size() +
             hidden_size * num_tokens + num_tokens * hidden_size//block_shape[0] * 4 +
             top_k * num_tokens * w1.shape[2] * 2)

    mem_2 = (activated_experts * w2.shape[1] * w2.shape[2] * w2.element_size() +
             activated_experts * w2.shape[1]//block_shape[0] * w2.shape[2]//block_shape[1] * w2_scale.element_size() +
             hidden_size * num_tokens + activated_experts * num_tokens * hidden_size//block_shape[0] * 4 +
             num_tokens * w2.shape[2] * 2)

    return flops_1, flops_2, mem_1, mem_2


def parse_arguments():
    parser = argparse.ArgumentParser(description='MOE Performance Test')

    parser.add_argument('--batch-sizes',
                        type=int,
                        nargs='+',
                        default=[8, 32, 128, 256, 512, 1024, 2048, 4096, 8192],
                        help='Batch sizes to benchmark')

    parser.add_argument('--balancedness',
                        type=float,
                        nargs='+',
                        default=[0.8, 0.7, 0.6, 0.5],
                        help='Balancedness values to test')

    parser.add_argument('--N',
                        type=int,
                        default=256,
                        help='N dimension for MoE')

    parser.add_argument('--K',
                        type=int,
                        default=7168,
                        help='K dimension for MoE')

    parser.add_argument('--E',
                        type=int,
                        default=256,
                        help='num_experts for MoE')

    parser.add_argument('--top-k',
                        type=int,
                        default=8,
                        help='top-k experts picked')

    parser.add_argument('--shared-expert',
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help='If MoE is using a shared expert')

    parser.add_argument('--config',
                        default=None,
                        help='Path to JIT config file, default: moe_jit_E_N_K.json')

    parser.add_argument('--scaling-factor',
                        type=float,
                        default=2.5,
                        help='Output scaling factor')

    return parser.parse_args()


torch.manual_seed(42)
torch.set_default_device("cuda:0")


if __name__ == "__main__":
    args = parse_arguments()

    hidden_size = args.K
    top_k = args.top_k + args.shared_expert
    block_shape = [128, 128]
    E = args.E + args.shared_expert
    N = args.N
    K = args.K

    w1 = torch.randn((E, N, K)).to(torch.float8_e4m3fn).to("cuda:0")
    w2 = torch.randn((E, K, N//2)).to(torch.float8_e4m3fn).to("cuda:0")

    w1_scale = torch.randn((E, w1.shape[1]//block_shape[0], w1.shape[2]//block_shape[1]), dtype=torch.float32) * 0.01
    w2_scale = torch.randn((E, w2.shape[1]//block_shape[0], w2.shape[2]//block_shape[1]), dtype=torch.float32) * 0.01

    batch_sizes = args.batch_sizes
    balancedness_values = args.balancedness
    config_path = args.config or f"moe_jit_{E}_{N}_{K}.json"

    for num_tokens in batch_sizes:
        print(f"\nBatch size {num_tokens}")
        topk_weights = torch.nn.functional.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32), dim=-1)
        config_dtype = 'fp8_w8a8'
        triton_config = try_get_optimal_moe_config(w1.shape, w2.shape, top_k, config_dtype, block_shape=block_shape, M=num_tokens)
        compute_type = tl.bfloat16

        for balancedness in balancedness_values:
            topk_ids = generate_topk_ids(E-1, num_tokens, top_k-1, balancedness)
            if args.shared_expert:
                topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens).view(num_tokens,1).to(torch.int32)*(E-1)))

            x = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16).normal_(mean=0, std=0.05)
            x_q, x_scale = sglang_per_token_group_quant_fp8(x, block_shape[1])

            local_conf = get_best_config(config_path, num_tokens)

            block_m = local_conf["block_m"]
            bn = local_conf["block_n"]
            wn = local_conf["warp_n"]
            stages = local_conf["stages"]

            sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(
                topk_ids, triton_config["BLOCK_SIZE_M"], E
            )
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_m, E)

            out_triton_up = torch.empty((num_tokens, top_k, w1.shape[1]), device=x.device, dtype=x.dtype)
            out_triton_swiglu = torch.empty((num_tokens*top_k, w1.shape[1]//2), device=x.device, dtype=x.dtype)
            out_triton_down = torch.empty((num_tokens, top_k, x.shape[1]), device=x.device, dtype=x.dtype)
            out_triton = torch.empty_like(x)

            triton_time_up = bench_kineto(
                lambda: invoke_fused_moe_kernel(
                    x, w1, None, out_triton_up, None, w1_scale, None, topk_weights, topk_ids,
                    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton,
                    False, top_k, triton_config, compute_type, True, False, False, False, False, block_shape
                )
            )

            triton_time_swiglu = bench_kineto(
                lambda: silu_and_mul(out_triton_up.view(-1, w1.shape[1]), out_triton_swiglu),
                kernel_name="silu"
            )

            triton_time_quant = bench_kineto(
                lambda: invoke_fused_moe_kernel(
                    out_triton_swiglu, w2, None, out_triton_down, None, w2_scale, None, topk_weights, topk_ids,
                    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton,
                    False, top_k, triton_config, compute_type, True, False, False, False, False, block_shape
                ),
                kernel_name="per_token_group_quant_8bit_kernel"
            )

            triton_time_down = bench_kineto(
                lambda: invoke_fused_moe_kernel(
                    out_triton_swiglu, w2, None, out_triton_down, None, w2_scale, None, topk_weights, topk_ids,
                    sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton,
                    False, 1, triton_config, compute_type, True, False, False, False, False, block_shape
                )
            )

            tokens_in_chunk = out_triton_swiglu.shape[0]
            if tokens_in_chunk < 32:
                triton_time_merge = bench_kineto(
                    lambda: moe_sum_reduce_torch_compile(out_triton_down.view(*out_triton_down.shape), out_triton, args.scaling_factor),
                    kernel_name="triton_per_fused_mul_sum_0"
                )
            else:
                triton_time_merge = bench_kineto(
                    lambda: moe_sum_reduce_triton(out_triton_down.view(*out_triton_down.shape), out_triton, args.scaling_factor),
                    kernel_name="sum_reduce"
                )

            triton_time = triton_time_merge + triton_time_down + triton_time_up + triton_time_swiglu + triton_time_quant

            out = torch.zeros_like(x)
            torch.ops.alpha_moe.fused_moe_w8a8_up_down(
                x_q, x_scale, w1, w1_scale, w2, w2_scale,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                topk_weights, out, top_k,
                block_m, bn, wn, stages, args.scaling_factor
            )

            alpha_time = bench_kineto(
                lambda: torch.ops.alpha_moe.fused_moe_w8a8_up_down(
                    x_q, x_scale, w1, w1_scale, w2, w2_scale,
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    topk_weights, out, top_k,
                    block_m, bn, wn, stages, args.scaling_factor
                )
            )

            activated_experts = len(set(topk_ids.flatten().tolist()))
            f1, f2, m1, m2 = get_stats(activated_experts, num_tokens, top_k, w1, w2, w1_scale, w2_scale, hidden_size, block_shape)
            total_flops = f1 + f2
            total_mem = m1 + m2

            print(f"  balancedness={balancedness:.2f}, block_m={block_m}, bn={bn}, wn={wn}, stages={stages}")
            print(f"    Triton: {triton_time:.2f} us, {(total_flops/1e6)/triton_time:.2f} TFLOPs, {(total_mem/1e3)/triton_time:.2f} GB/s")
            print(f"    Alpha:  {alpha_time:.2f} us, {(total_flops/1e6)/alpha_time:.2f} TFLOPs, {(total_mem/1e3)/alpha_time:.2f} GB/s")
            print(f"    Speedup: {triton_time/alpha_time:.2f}x")
