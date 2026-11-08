"""Binomial shortmer encoding exactly as in Preuss, Yakhini, Anavy (2021), Sec. 2.1-2.3.

Fixed paper parameters (no free knobs):
  Omega  = 16 trimers of Table 1 (Hamming distance d = 2)
  K      = 5 trimers per letter  ->  C(16,5) = 4368 letters, first 4096 used
  12 bits per letter, barcode = 12 standard nucleotides, payload = 120 letters per strand
"""
from itertools import combinations
import numpy as np

OMEGA = ["AAT", "ACA", "ATG", "AGC", "TAA", "TCT", "TTC", "TGG",
         "GAG", "GCC", "GTT", "GGA", "CAC", "CCG", "CTA", "CGT"]      # Table 1: X1..X16
N, K, BITS = 16, 5, 12
LETTERS_PER_STRAND, BARCODE_LEN = 120, 12

# letter index -> the 5 trimer indices (lexicographic order of 5-subsets of {0..15})
SUBSETS = np.array(list(combinations(range(N), K))[: 1 << BITS], dtype=np.int8)   # (4096, 5)
_MASK = (1 << SUBSETS.astype(np.int32)).sum(axis=1)                                 # 16-bit mask
MASK_TO_LETTER = np.full(1 << N, -1, dtype=np.int32)
MASK_TO_LETTER[_MASK] = np.arange(1 << BITS)
_SHIFTS = np.arange(BITS - 1, -1, -1)


def bytes_to_letters(data: bytes) -> np.ndarray:
    """file -> bits -> groups of 12 bits -> letter numbers 0..4095 (padded to whole strands)."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    per_strand = BITS * LETTERS_PER_STRAND
    bits = np.pad(bits, (0, -len(bits) % per_strand))
    return (bits.reshape(-1, BITS).astype(np.int32) << _SHIFTS).sum(axis=1)


def letters_to_bytes(letters: np.ndarray, n_bytes: int) -> bytes:
    bits = ((letters[:, None] >> _SHIFTS) & 1).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes()[:n_bytes]


def barcode(i: int) -> str:
    """Standard-DNA barcode: strand number written in base 4 (A,C,G,T)."""
    return "".join("ACGT"[(i >> (2 * s)) & 3] for s in range(BARCODE_LEN - 1, -1, -1))


def letter_trimers(letter: int) -> list[str]:
    return [OMEGA[j] for j in SUBSETS[letter]]


def simulate(letters: np.ndarray, reads: int, p_err: float, seed: int = 1, chunk: int = 400_000):
    """Read the combinatorial letters back, with and without substitution errors.

    Every letter is a mixture of its 5 trimers. A read of one position returns one trimer chosen
    at random (coupon collector, paper Sec. 2.3). With probability p_err a read has ONE wrong base;
    because d = 2 this never produces another member of Omega (paper: it is detected).
    Returns decoded letter arrays for three decoders (unreadable letters -> 0):
      clean   : no errors, just the random sampling of `reads` reads per letter
      plain   : with errors, decoder accepts the raw set of trimers it sees
      checked : with errors, decoder first throws away reads that are not in Omega (d = 2 check)
    """
    rng = np.random.default_rng(seed)
    out = {k: np.zeros_like(letters) for k in ("clean", "plain", "checked")}
    for s in range(0, len(letters), chunk):
        L = letters[s:s + chunk]
        counts = rng.multinomial(reads, [1 / K] * K, size=len(L))     # reads of each of the 5 trimers
        bad = rng.binomial(counts, p_err)                              # of those, reads with 1 wrong base
        all_seen = (counts > 0).all(axis=1)
        no_err = bad.sum(axis=1) == 0
        clean_seen = ((counts - bad) > 0).all(axis=1)
        out["clean"][s:s + chunk] = np.where(all_seen, L, 0)
        out["plain"][s:s + chunk] = np.where(all_seen & no_err, L, 0)
        out["checked"][s:s + chunk] = np.where(clean_seen, L, 0)
    return out


def example_molecules(letter_row: np.ndarray, n: int = 6, seed: int = 3) -> list[str]:
    """A few of the physical molecules of one strand: each picks one trimer per letter at random."""
    rng = np.random.default_rng(seed)
    return ["".join(OMEGA[SUBSETS[l][rng.integers(K)]] for l in letter_row) for _ in range(n)]
