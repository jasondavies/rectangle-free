from math import comb
import sys


def falling(n: int, m: int) -> int:
    p = 1
    for i in range(m):
        p *= n - i
    return p


def rectfree_2xn_4(n: int) -> int:
    fall4 = [1, 4, 12, 24, 24]  # (4)_s for s=0..4
    return sum(comb(n, s) * fall4[s] * (12 ** (n - s)) for s in range(min(4, n) + 1))


def poly_mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def poly_pow(p: list[int], k: int) -> list[int]:
    out = [1]
    base = p[:]
    e = k
    while e:
        if e & 1:
            out = poly_mul(out, base)
        e >>= 1
        if e:
            base = poly_mul(base, base)
    return out


def rectfree_3xn_k(n: int, k: int) -> int:
    if n < 0 or k < 0:
        return 0
    km1 = k - 1
    g = [1, 1 + 3 * km1, 3 * (km1**2), km1**3]
    s = poly_pow(g, k)
    free_columns = k * (k - 1) * (k - 2)
    total = 0
    for m in range(min(len(s) - 1, n) + 1):
        total += s[m] * falling(n, m) * (free_columns ** (n - m))
    return total


def rectfree_count4(rows: int, n: int) -> int:
    if rows == 2:
        return rectfree_2xn_4(n)
    if rows == 3:
        return rectfree_3xn_k(n, 4)
    raise ValueError(f"unsupported row count for count4.py: {rows}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 count4.py ROWS")
    rows = int(sys.argv[1])
    if rows not in (2, 3):
        raise SystemExit(f"count4.py currently supports ROWS=2 or ROWS=3, got {rows}")
    for n in range(1, 41):
        print(n, rectfree_count4(rows, n))


if __name__ == "__main__":
    main()
