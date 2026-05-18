import math


def safe_list_get(lst, idx, default=None):
    """Return lst[idx] or default on IndexError."""
    try:
        return lst[idx]
    except IndexError:
        return default


def cosine_ease_in_out_minmax(cur, max_step):
    x = min(cur / (max_step + 1e-8), 1)
    return 0.5 * (1 - math.cos(x * math.pi))
