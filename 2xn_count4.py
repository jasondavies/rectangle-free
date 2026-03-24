from math import comb

def rectfree_2xn_4(n: int) -> int:
    fall4 = [1, 4, 12, 24, 24]  # (4)_s for s=0..4
    return sum(comb(n, s) * fall4[s] * (12 ** (n - s)) for s in range(min(4, n) + 1))

for n in range(1, 41):
    print(n, rectfree_2xn_4(n))
