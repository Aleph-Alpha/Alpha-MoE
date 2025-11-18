import torch
import alpha_moe


@torch.library.register_fake("alpha_moe::fused_moe_w8a8_up_down")
def fused_moe_w8a8_up_down_abstract(
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
    return out


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
    return torch.ops.alpha_moe.fused_moe_w8a8_up_down(
        x,
        x_scale,
        w,
        w_scale,
        w2,
        w2_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        block_n,
        warp_n,
        stages,
        scaling_factor,
    )


__all__ = ["fused_moe_w8a8_up_down"]
