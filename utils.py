import json
import functools

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

