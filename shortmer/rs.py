"""Reed-Solomon codes over GF(2^m) with errors-and-erasures decoding.

The paper (Preuss, Yakhini, Anavy) uses RS codes over GF(2^4) for barcodes and
over GF(2^9..2^13) for payloads, which is larger than the GF(2^8) supported by
common packages, so a small generic implementation is provided here.

Conventions
-----------
* A codeword is a list of field symbols, index 0 = highest-degree coefficient.
* Systematic: codeword = message + parity.
* Generator roots are alpha^fcr ... alpha^(fcr+nsym-1).
* Polynomials used internally are lists with the *lowest* degree first.
"""
from __future__ import annotations

# Primitive polynomials (including the x^m term) for GF(2^m).
PRIMITIVE_POLY = {
    2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011,
    7: 0b10001001, 8: 0x11D, 9: 0x211, 10: 0x409, 11: 0x805, 12: 0x1053,
    13: 0x201B, 14: 0x4443, 15: 0x8003, 16: 0x1100B,
}

_FIELDS: dict[int, "GF"] = {}


class GF:
    """Finite field GF(2^m) implemented with log / antilog tables."""

    def __init__(self, m: int):
        if m not in PRIMITIVE_POLY:
            raise ValueError(f"GF(2^{m}) is not supported (2 <= m <= 16)")
        self.m = m
        self.q = 1 << m
        self.order = self.q - 1
        poly = PRIMITIVE_POLY[m]
        self.exp = [0] * (2 * self.order)
        self.log = [0] * self.q
        x = 1
        for i in range(self.order):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.q:
                x ^= poly
        if x != 1:  # the polynomial must be primitive
            raise ValueError(f"polynomial for m={m} is not primitive")
        for i in range(self.order, 2 * self.order):
            self.exp[i] = self.exp[i - self.order]

    @classmethod
    def get(cls, m: int) -> "GF":
        if m not in _FIELDS:
            _FIELDS[m] = cls(m)
        return _FIELDS[m]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError
        if a == 0:
            return 0
        return self.exp[self.log[a] - self.log[b] + self.order]

    def pow_alpha(self, e: int) -> int:
        return self.exp[e % self.order]


# --- polynomial helpers (lowest degree first) -------------------------------
def _pmul(gf: GF, a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    out[i + j] ^= gf.mul(ai, bj)
    return out


def _peval(gf: GF, p: list[int], x: int) -> int:
    y = 0
    for c in reversed(p):
        y = gf.mul(y, x) ^ c
    return y


def _berlekamp_massey(gf: GF, s: list[int]) -> list[int]:
    """Shortest LFSR (connection polynomial, lowest-first) generating s."""
    C, B = [1], [1]
    L, m, b = 0, 1, 1
    for n in range(len(s)):
        d = s[n]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= gf.mul(C[i], s[n - i])
        if d == 0:
            m += 1
            continue
        coef = gf.div(d, b)
        T = C[:]
        need = len(B) + m
        if len(C) < need:
            C = C + [0] * (need - len(C))
        for i, bi in enumerate(B):
            C[i + m] ^= gf.mul(coef, bi)
        if 2 * L <= n:
            L, B, b, m = n + 1 - L, T, d, 1
        else:
            m += 1
    C = C[: L + 1] + [0] * max(0, L + 1 - len(C))
    return C


class RSCodec:
    """Systematic RS(n, n - nsym) codec over GF(2^m), n <= 2^m - 1."""

    def __init__(self, m: int, nsym: int, fcr: int = 1):
        self.gf = GF.get(m)
        self.m = m
        self.nsym = nsym
        self.fcr = fcr
        gf = self.gf
        g = [1]
        for i in range(nsym):
            g = _pmul(gf, g, [gf.pow_alpha(fcr + i), 1])
        # coefficients of x^(nsym-1) ... x^0 (monic leading term dropped)
        self._gen_hi = list(reversed(g))[1:]

    # -- encoding -----------------------------------------------------------
    def parity(self, msg: list[int]) -> list[int]:
        if len(msg) + self.nsym > self.gf.order:
            raise ValueError("codeword longer than 2^m - 1")
        if self.nsym == 0:
            return []
        gf, rem = self.gf, [0] * self.nsym
        for s in msg:
            fb = s ^ rem[0]
            rem = rem[1:] + [0]
            if fb:
                for i, gi in enumerate(self._gen_hi):
                    if gi:
                        rem[i] ^= gf.mul(fb, gi)
        return rem

    def encode(self, msg: list[int]) -> list[int]:
        return list(msg) + self.parity(msg)

    # -- decoding -----------------------------------------------------------
    def syndromes(self, word: list[int]) -> list[int]:
        gf = self.gf
        out = []
        for i in range(self.nsym):
            a = gf.pow_alpha(self.fcr + i)
            y = 0
            for sym in word:
                y = gf.mul(y, a) ^ sym
            out.append(y)
        return out

    def decode(self, word: list[int], erasures=()) -> tuple[list[int] | None, int]:
        """Correct up to t errors and e erasures with 2t + e <= nsym.

        Returns (corrected_codeword, n_changed_symbols) or (None, 0) on
        decoding failure.
        """
        gf, n, nsym = self.gf, len(word), self.nsym
        if n > gf.order:
            raise ValueError("codeword longer than 2^m - 1")
        erasures = sorted(set(erasures))
        if len(erasures) > nsym:
            return None, 0
        w = list(word)
        for j in erasures:
            w[j] = 0
        S = self.syndromes(w)
        if not any(S):
            return w, sum(1 for a, b in zip(w, word) if a != b)

        # erasure locator  Gamma(x) = prod (1 - X_j x)
        gamma = [1]
        for j in erasures:
            gamma = _pmul(gf, gamma, [1, gf.pow_alpha(n - 1 - j)])
        nu = len(erasures)
        # Forney (modified) syndromes
        T = _pmul(gf, S, gamma)[:nsym]
        T += [0] * (nsym - len(T))
        lam = _berlekamp_massey(gf, T[nu:])
        n_err = len(lam) - 1
        if 2 * n_err + nu > nsym:
            return None, 0
        psi = _pmul(gf, lam, gamma)
        while len(psi) > 1 and psi[-1] == 0:
            psi.pop()
        deg = len(psi) - 1

        positions = []
        for j in range(n):
            xinv = gf.pow_alpha(-(n - 1 - j))
            if _peval(gf, psi, xinv) == 0:
                positions.append(j)
        if len(positions) != deg:
            return None, 0

        omega = _pmul(gf, S, psi)[:nsym]
        dpsi = [psi[i] if i % 2 == 1 else 0 for i in range(1, len(psi))]
        for j in positions:
            x = gf.pow_alpha(n - 1 - j)
            xinv = gf.pow_alpha(-(n - 1 - j))
            den = _peval(gf, dpsi, xinv)
            if den == 0:
                return None, 0
            mag = gf.div(_peval(gf, omega, xinv), den)
            mag = gf.mul(mag, gf.pow_alpha((n - 1 - j) * (1 - self.fcr)))
            w[j] ^= mag
        if any(self.syndromes(w)):
            return None, 0
        return w, sum(1 for a, b in zip(w, word) if a != b)
