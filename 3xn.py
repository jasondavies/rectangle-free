from math import comb

def falling(n, m):
    p = 1
    for i in range(m):
        p *= (n - i)
    return p

def poly_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out

def poly_pow(p, k):
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
    # g_k(x) = 1 + (1+3(k-1)) x + 3(k-1)^2 x^2 + (k-1)^3 x^3
    km1 = k - 1
    g = [1,
         1 + 3*km1,
         3*(km1**2),
         (km1**3)]
    # G_k(x) = g^k, coefficients s_m
    s = poly_pow(g, k)

    A = k*(k-1)*(k-2)  # free columns
    total = 0
    for m in range(min(len(s)-1, n) + 1):
        total += s[m] * falling(n, m) * (A ** (n - m))
    return total

for n in range(1, 41):
    print(n, rectfree_3xn_k(n,4))
