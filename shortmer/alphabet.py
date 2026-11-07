"""k-mer set Omega and the binomial combinatorial alphabet Sigma (paper Sec. 2.1-2.2)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, floor, log2

NT = "ACGT"

# Paper, Table 4: trimer set Omega_1, |Omega_1| = 16, minimum Hamming distance 2.
PAPER_OMEGA_1 = ["AAT", "ACA", "ATG", "AGC", "TAA", "TCT", "TTC", "TGG",
                 "GAG", "GCC", "GTT", "GGA", "CAC", "CCG", "CTA", "CGT"]


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def min_hamming(kmers: list[str]) -> int:
    if len(kmers) < 2:
        return len(kmers[0]) if kmers else 0
    return min(hamming(a, b) for i, a in enumerate(kmers) for b in kmers[i + 1:])


def build_omega(k: int, n: int, d: int) -> list[str]:
    """Return n k-mers with pairwise Hamming distance >= d.

    For the paper's setting (k=3, d=2, n<=16) the published set Omega_1 is
    used; otherwise a deterministic greedy (lexicographic) search is run.
    """
    if k == 3 and d == 2 and n <= len(PAPER_OMEGA_1):
        return PAPER_OMEGA_1[:n]
    chosen: list[str] = []
    for tup in product(NT, repeat=k):
        w = "".join(tup)
        if all(hamming(w, c) >= d for c in chosen):
            chosen.append(w)
            if len(chosen) == n:
                return chosen
    raise ValueError(
        f"Only {len(chosen)} {k}-mers with minimum Hamming distance {d} exist "
        f"(greedy search); requested N={n}.")


def unrank(idx: int, n: int, k: int) -> tuple[int, ...]:
    """idx-th K-subset of {0..n-1} in lexicographic order."""
    out, x = [], 0
    for i in range(k):
        while True:
            c = comb(n - x - 1, k - i - 1)
            if idx < c:
                out.append(x)
                x += 1
                break
            idx -= c
            x += 1
    return tuple(out)


def rank(subset: tuple[int, ...], n: int, k: int) -> int:
    """Inverse of :func:`unrank`."""
    idx, prev = 0, -1
    for i, c in enumerate(subset):
        for x in range(prev + 1, c):
            idx += comb(n - x - 1, k - i - 1)
        prev = c
    return idx


@dataclass
class Alphabet:
    """Binomial alphabet: letters are K-subsets of the N k-mers in Omega."""

    omega: list[str]
    K: int

    def __post_init__(self):
        self.N = len(self.omega)
        self.k = len(self.omega[0])
        if not 1 <= self.K < self.N:
            raise ValueError("need 1 <= K < N")
        self.size = comb(self.N, self.K)
        self.bits = floor(log2(self.size))          # bits stored per letter
        self.usable = 1 << self.bits                # letters actually used
        if self.bits > 16:
            raise ValueError("more than 16 bits per letter is not supported")
        if self.bits < 2:
            raise ValueError("alphabet too small (< 4 letters)")
        self.min_distance = min_hamming(self.omega)

    # letter <-> subset ------------------------------------------------------
    def subset(self, letter: int) -> tuple[int, ...]:
        """Indices (0-based) into Omega of the K k-mers forming a letter."""
        return unrank(letter, self.N, self.K)

    def kmers(self, letter: int) -> list[str]:
        return [self.omega[i] for i in self.subset(letter)]

    def letter(self, subset: tuple[int, ...]) -> int | None:
        r = rank(tuple(sorted(subset)), self.N, self.K)
        return r if r < self.usable else None

    @property
    def min_distance_ok(self) -> bool:
        return self.min_distance >= 2
