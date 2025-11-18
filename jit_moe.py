import torch
import json
import argparse
from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
        )
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        moe_align_block_size,
        )
from statistics import mean
import alpha_moe


# Adapted from: https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/utils.py
def bench_kineto(fn, kernel_name: str = "moe", num_tests: int = 25):
    # Profile
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
        for i in range(2):
            for _ in range(num_tests):
                # Alocate big tensor to clear cache
                x = torch.randn((8, 8192, 8192), dtype=torch.float, device='cuda')
                fn()
                torch.cuda.synchronize()
            prof.step()

    # Parse the profiling table
    times = []
    for e in prof.profiler.function_events:
        if kernel_name in e.name:
            times.append(e.device_time_total)
    return sum(times)/len(times)

def generate_topk_ids(num_experts, num_tokens, top_k, balancedness=1.0):
    """
    Generate topk_ids with a given balancedness.

    balancedness:
        1.0 -> perfectly balanced (uniform)
        0.0 -> maximally skewed (all tokens go to one expert)
        in between -> mixture
    """
    # interpolate between uniform and skewed distribution
    uniform = torch.ones(num_experts) / num_experts
    skewed = torch.zeros(num_experts); skewed[0] = 1.0
    probs = balancedness * uniform + (1 - balancedness) * skewed

    # sample expert assignments
    topk_ids = torch.multinomial(probs, num_tokens * top_k, replacement=True)
    topk_ids = topk_ids.view(num_tokens, top_k)
    return topk_ids


def parse_arguments():
    parser = argparse.ArgumentParser(description='MOE Benchmark Script')

    parser.add_argument('--batch-sizes',
                        type=int,
                        nargs='+',
                        default=[8, 32, 128, 256, 512, 1024, 2048, 4096, 8192],
                        help='Batch sizes to benchmark')

    parser.add_argument('--balancedness',
                        type=float,
                        nargs='+',
                        default=[0.8, 0.7, 0.6, 0.5],
                        help='Balancedness values to test ')

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
                        help='num_experts for MoE(not including shared expert)')

    parser.add_argument('--top-k',
                        type=int,
                        default=8,
                        help='top-k experts picked')

    parser.add_argument('--shared-expert',
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help='If MoE is using a shared expert')

    parser.add_argument('--out-config',
                        default=None,
                        help='Where to store config, default: moe_jit_E_N_K.json')

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
    times = {}

    for num_tokens in batch_sizes:
        entries = {}
        for block_m in range(8, 129, 8):
            for bn, wn in [(64, 4), (32, 8)]:
                for stages in range(1, 6):
                    entries[(block_m, bn, wn, stages)] = []
        times[num_tokens] = entries
        print("Batch size", num_tokens)
        topk_weights = torch.nn.functional.softmax(torch.randn((num_tokens, top_k), dtype=torch.float32), dim=-1)

        all_topks = []
        for balancedness in balancedness_values:
            topk_ids = generate_topk_ids(E-1, num_tokens, top_k-1)
            if args.shared_expert:
                topk_ids = torch.hstack((topk_ids, torch.ones(num_tokens).view(num_tokens,1).to(torch.int32)*(E-1)))
            all_topks.append(topk_ids)

        for topk_ids in all_topks:
            x = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16).normal_(mean=0, std=0.05)
            x_q, x_scale = sglang_per_token_group_quant_fp8(x, block_shape[1])
            x_sc = x_scale.repeat_interleave(block_shape[0], 1)

            bench_fn = bench_kineto

            best_configuration = ""
            best_diff = (-1, -1)
            best_time = float("inf")
            best_d_max = (-1, -1)
            for block_m in range(8, 129, 8):
                for bn, wn in [(64, 4), (32, 8)]:
                    for stages in range(1, 6):
                        if num_tokens < block_m and block_m != 16:
                            continue
                        if stages == 5 and block_m > 100:
                            continue
                        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_m, E)
                        configuration = f"{block_m=} {bn=}, {wn=}, {stages=}"
                        out = torch.zeros_like(x)
                        torch.ops.alpha_moe.fused_moe_w8a8_up_down(x_q, x_scale, w1, w1_scale, w2, w2_scale, sorted_token_ids,
                                                                      expert_ids, num_tokens_post_padded, topk_weights, out, top_k,
                                                                      block_m, bn, wn, stages, 2.5)

                        new_time = bench_fn(lambda: torch.ops.alpha_moe.fused_moe_w8a8_up_down(x_q, x_scale, w1, w1_scale, w2, w2_scale, sorted_token_ids,
                                                                                                  expert_ids, num_tokens_post_padded, topk_weights, out, top_k,
                                                                                                  block_m, bn, wn, stages, 2.5))
                        times[num_tokens][(block_m, bn, wn, stages)].append(new_time)
    conf_to_save = {}
    for nt, val in times.items():
        t = [(k, mean(v)) for k, v in val.items() if len(v) > 0]
        best_configuration = sorted(t, key=lambda x: x[1])[0]
        (block_m, bn, wn, stages) = best_configuration[0]
        conf_to_save[nt] = {
                "block_m": block_m,
                "block_n": bn,
                "warp_n": wn,
                "stages": stages
                }

    out_config = args.out_config or f"moe_jit_{E}_{N}_{K}.json"
    with open(out_config, "w") as f:
        json.dump(conf_to_save, f)


