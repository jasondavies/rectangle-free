import itertools
from collections import Counter
from functools import lru_cache
from math import comb, factorial

ROWS = 4
COLORS = 4

pairs = [(i, j) for i in range(ROWS) for j in range(i + 1, ROWS)]
pair_index = {p: k for k, p in enumerate(pairs)}  # 6 pairs

def col_mask(col):
    """24-bit mask: bit (colour*6 + pair) is 1 if that row-pair is monochromatic in that colour."""
    m = 0
    for (i, j), pi in pair_index.items():
        if col[i] == col[j]:
            c = col[i]
            m |= 1 << (c * 6 + pi)
    return m

# Count how many column-types yield each nonzero mask; count free columns too.
w = Counter()
free = 0
for col in itertools.product(range(COLORS), repeat=ROWS):
    m = col_mask(col)
    if m == 0:
        free += 1
    else:
        w[m] += 1

assert free == 24  # all 4 entries distinct

items = list(w.items())  # (mask, weight)
perbit = [[] for _ in range(24)]
for mask, wt in items:
    for b in range(24):
        if (mask >> b) & 1:
            perbit[b].append((mask, wt))
for b in range(24):
    perbit[b].sort(key=lambda x: (-x[0].bit_count(), x[0]))

U0 = (1 << 24) - 1

def poly_add(a, b):
    if len(b) > len(a):
        a, b = b, a
    res = list(a)
    for i, v in enumerate(b):
        res[i] += v
    return res

def poly_shift_mul(a, wt):
    # multiply by (wt * x)
    return [0] + [wt * v for v in a]

@lru_cache(maxsize=None)
def solve(U):
    """Return poly p where p[k] = sum of products of weights over all k-mask selections within U."""
    if U == 0:
        return (1,)
    lsb = U & -U
    b = lsb.bit_length() - 1

    # option: bit b unused
    p = list(solve(U ^ lsb))

    # option: choose exactly one item containing b
    for mask, wt in perbit[b]:
        if mask & ~U:
            continue
        q = list(solve(U & ~mask))
        q = poly_shift_mul(q, wt)
        p = poly_add(p, q)
    return tuple(p)

c = solve(U0)                 # c[k] = weight-sum over k chosen masks (unordered)
B = [factorial(k) * c[k] for k in range(len(c))]  # ordered sequences length k

def F(n: int) -> int:
    return sum(comb(n, k) * (free ** (n - k)) * B[k] for k in range(min(n, 24) + 1))

# demo
for n in range(1, 41):
    print(n, F(n))
