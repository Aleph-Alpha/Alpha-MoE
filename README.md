This is a package containing a code for a fused Mixture of Experts megakernel tailored for TP servings of MoE models. It provides an interface
that is compatible with how MoE layers are implemented inside vLLM and SGLang

# Installation

```
git clone https://github.com/Aleph-Alpha/Alpha-MoE.git
cd Alpha-MoE
pip install -e . --no-build-isolation
```

# Usage

##
Alpha MoE provides torch bindings as well as an interface for them inside `alpha_moe_ops.py` 

```Py
def fused_moe_w8a8_up_down(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    block_n: int,
    warp_n: int,
    stages: int,
    scaling_factor: float,
) -> torch.Tensor:
    """
    Args:
        x: Input tensor [M, K] in FP8 format
        x_scale: Per-token scale factors for x
        w: First weight matrix [num_experts, N, K] in FP8 format (up projection)
        w_scale: Scale factors for w
        w2: Second weight matrix [num_experts, K, N//2] in FP8 format (down projection)
        w2_scale: Scale factors for w2
        sorted_token_ids: Sorted token indices
        expert_ids: Expert indices for each block
        num_tokens_post_padded: Number of tokens after padding
        topk_weights: Weights for top-k experts per token
        out: Pre-allocated output tensor [M, K] in BF16 format (modified in-place, must be initialized to zeros)
        top_k: Number of top experts per token
        block_m: Tuning parameter: Block size in M dimension
        block_n: Tuning parameter: Block size in N dimension
        warp_n: Tuning parameter: Number of warps in N dimension
        stages: Tuning parameter: Pipeline stages
        scaling_factor: Scaling factor for the output

    Returns:
        Output tensor (same as `out` parameter) - modified in-place
    """
```

Note that the output tensor must be zero initialized

Parameters `block_m`, `block_n`, `warp_n` and `stages` are configuration parameters that influence the speed of the kernel. Fused-MoE provides a script
that can search them for you.

## Finding optimal configurations

For finding optimal configurations you can use `jit_moe.py` script with the shapes your model uses:

```
python jit_moe.py --E 512 --N 256 --K 2048 --no-shared-expert --out-file moe_jit.json
```

For performance comparisons against SGLang you can later run `test/test_moe_performance.py` with the same parameters

We also provide an utils that can help find the optimal configuration for current num_input_tokens. With the script installed the invocation can look like this:

```
M = num_tokens
local_conf = get_best_config(os.getenv("ALPHA_MOE_CONFIG"), M)
block_m = local_conf["block_m"]
bn = local_conf["block_n"]
wn = local_conf["warp_n"]
stages = local_conf["stages"]
A, A_scale = sglang_per_token_group_quant_fp8(hidden_states, block_shape[1])
hidden_states.zero_()

torch.ops.alpha_moe.fused_moe_w8a8_up_down(A, A_scale, w1, w1_scale, w2, w2_scale, sorted_token_ids,
                                                 expert_ids, num_tokens_post_padded, topk_weights, hidden_states,
                                                 topk, block_m, bn, wn, stages, routed_scaling_factor)
```

## Usage inside SGLang

For sglang we provide a patch file inside extra. After pulling the official SGLang docker container and installing Alpha-MoE you can 
```
git apply AlphaMoe/extra/sglang.patch
```

And run the MoE model with 

```
ALPHA_MOE_CONFIG=moe_jit.json python -m sglang.launch_server ...
```
