import json
import functools
import torch

def interleave_tensor(tensor, rep=8):
    M, N, K = tensor.shape

    first_half = tensor[:, :(N//2), :]
    second_half = tensor[:, (N//2):, :]

    first_chunks = first_half.view(M, (N//(2*rep)), rep, K)
    second_chunks = second_half.view(M, (N//(2*rep)), rep, K)

    interleaved = torch.stack([first_chunks, second_chunks], dim=2)
    result = interleaved.view(M, N, K)

    return result.contiguous()

@functools.lru_cache()
def get_best_config(path: str, n_tokens: int):
    with open(path, "r") as f:
        best_conf = json.load(f)
    dist = float("inf")
    ret = None
    for nt, val in best_conf.items():
        if abs(int(nt) - n_tokens) < dist:
            dist = abs(int(nt) - n_tokens)
            ret = val
    return ret

