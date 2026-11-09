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


def barcode_digits(ids: np.ndarray) -> np.ndarray:
    """Strand number in base 4 (11 nt) + 1 parity nt (sum of digits mod 4). Two valid barcodes always differ in at
    least 2 positions, so ONE substitution in a barcode can never give another valid barcode."""
    ids = np.asarray(ids)[..., None]
    dig = ((ids >> (2 * np.arange(BARCODE_LEN - 2, -1, -1))) & 3).astype(np.uint8)
    return np.concatenate([dig, (dig.sum(axis=-1, keepdims=True) % 4).astype(np.uint8)], axis=-1)


def barcode(i: int) -> str:
    return "".join("ACGT"[d] for d in barcode_digits(np.array(i)))


def letter_trimers(letter: int) -> list[str]:
    return [OMEGA[j] for j in SUBSETS[letter]]



# ----------------------------------------------------------------------------- molecules / FASTQ
LPS = LETTERS_PER_STRAND
READ_LEN = BARCODE_LEN + 3 * LPS
ASCII = np.frombuffer(b"ACGT", dtype=np.uint8)
TRI_BASES = np.array([["ACGT".index(c) for c in t] for t in OMEGA], dtype=np.uint8)   # (16, 3)


def n_strands_for(n_bytes: int) -> int:
    return -(-n_bytes * 8 // (BITS * LPS))


def miss_probability(copies: int) -> float:
    """P(at least one of the 5 trimers of a letter is never seen in `copies` molecules) - coupon collector."""
    from math import comb
    return float(sum((-1) ** (j + 1) * comb(K, j) * (1 - j / K) ** copies for j in range(1, K + 1)))


def make_reads(letters: np.ndarray, copies: int, p_sub: float = 0.0, seed: int = 1):
    """Synthesize `copies` molecules per strand: every molecule takes ONE random trimer of the letter's 5 at each
    position (optionally with substitutions, see add_substitutions).
    Returns (reads as ASCII matrix [n_strands*copies, READ_LEN], strand id of every read)."""
    rng = np.random.default_rng(seed)
    n = len(letters) // LPS
    sub = SUBSETS[letters.reshape(n, LPS)]                                        # (n, 120, 5)
    pick = rng.integers(0, K, size=(n, copies, LPS, 1))
    tri = np.take_along_axis(np.broadcast_to(sub[:, None], (n, copies, LPS, K)), pick, axis=3)[..., 0]
    pay = TRI_BASES[tri].reshape(n, copies, 3 * LPS)
    bc = barcode_digits(np.arange(n))                                             # (n, 12)
    seq = np.concatenate([np.broadcast_to(bc[:, None, :], (n, copies, BARCODE_LEN)), pay], axis=2)
    seq = np.ascontiguousarray(seq.reshape(n * copies, READ_LEN), dtype=np.uint8)
    flat = seq.reshape(-1)
    n_err = rng.binomial(flat.size, p_sub)
    if n_err:
        pos = rng.integers(0, flat.size, n_err)
        flat[pos] = (flat[pos] + rng.integers(1, 4, n_err)) % 4                    # substitution only
    return ASCII[seq], np.repeat(np.arange(n), copies)


_IDX = np.full(256, 0, np.uint8)
for _i, _ch in enumerate(b"ACGT"):
    _IDX[_ch] = _i


def add_substitutions(reads: np.ndarray, p_sub: float, seed: int = 1):
    """Each nucleotide is replaced by a DIFFERENT base with probability p_sub. Returns (new reads, bases changed)."""
    out = reads.copy()
    if p_sub <= 0:
        return out, 0
    rng = np.random.default_rng(seed + 12345)
    flat = out.reshape(-1)
    n_err = int(rng.binomial(flat.size, p_sub))
    pos = np.unique(rng.integers(0, flat.size, n_err))
    flat[pos] = ASCII[(_IDX[flat[pos]] + rng.integers(1, 4, len(pos))) % 4]
    return out, len(pos)


def spaced(reads: np.ndarray) -> np.ndarray:
    """barcode + space + trimers separated by spaces (for reading by eye / CSV)."""
    m = len(reads)
    bc = reads[:, :BARCODE_LEN]
    tri = np.concatenate([reads[:, BARCODE_LEN:].reshape(m, LPS, 3), np.full((m, LPS, 1), 32, np.uint8)], axis=2)
    return np.concatenate([bc, np.full((m, 1), 32, np.uint8), tri.reshape(m, 4 * LPS)[:, :-1]], axis=1)


def to_fastq(reads: np.ndarray, copies: int, meta: dict) -> bytes:
    from urllib.parse import quote
    mt = " ".join(f"{k}={quote(str(v))}" for k, v in meta.items()).encode()
    qual = b"I" * READ_LEN
    return b"".join(b"@S%06d_C%03d %s\n%s\n+\n%s\n" % (k // copies, k % copies, mt, reads[k].tobytes(), qual)
                    for k in range(len(reads)))


def to_molecule_csv(reads: np.ndarray, copies: int) -> bytes:
    sp = spaced(reads)
    return b"strand,copy,barcode_and_trimers,sequence\n" + b"".join(
        b"%d,%d,%s,%s\n" % (k // copies, k % copies, sp[k].tobytes(), reads[k].tobytes()) for k in range(len(reads)))


def to_letter_csv(letters: np.ndarray) -> bytes:
    rows = ["strand,position,bits_12,letter_number,trimers,omega_indices"]
    for i, v in enumerate(letters.tolist()):
        rows.append(f"{i // LPS},{i % LPS},{v:012b},{v},{' '.join(letter_trimers(v))},"
                    f"{' '.join('X%d' % (j + 1) for j in SUBSETS[v])}")
    return "\n".join(rows).encode()


def parse_fastq(raw: bytes):
    """-> (reads ASCII matrix, metadata dict, number of reads discarded because their length is wrong)."""
    import gzip, re
    from urllib.parse import unquote
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    lines = raw.replace(b"\r", b"").split(b"\n")
    seqs = [x.strip().upper() for x in lines[1::4]]
    good = [x for x in seqs if len(x) == READ_LEN]
    meta = {k.decode(): unquote(v.decode()) for k, v in re.findall(rb"(\w+)=(\S+)", lines[0])} if lines else {}
    if "bytes" in meta:
        meta["bytes"] = int(meta["bytes"])
    reads = np.frombuffer(b"".join(good), dtype=np.uint8).reshape(-1, READ_LEN) if good else np.zeros((0, READ_LEN), np.uint8)
    return reads, meta, len(seqs) - len(good)



# ----------------------------------------------------------------------------- Reed-Solomon across strands (paper step ii)
# Groups of up to 42 data strands get 6 parity strands. Every column (letter position) is a codeword of a
# systematic Cauchy Reed-Solomon code over GF(2^12): any 6 unreadable letters of the 48 can be rebuilt.
RS_K, RS_P = 42, 6
_EXP = [0] * 8190
_LOG = [0] * 4096
_x = 1
for _i in range(4095):
    _EXP[_i], _LOG[_x] = _x, _i
    _x <<= 1
    if _x & 0x1000:
        _x ^= 0x1053                                  # x^12 + x^6 + x^4 + x + 1 (primitive)
_EXP[4095:] = _EXP[:4095]
_EXPA, _LOGA = np.array(_EXP, np.int32), np.array(_LOG, np.int32)


def gf_mul(a, b):
    a, b = np.asarray(a, np.int32), np.asarray(b, np.int32)
    return np.where((a == 0) | (b == 0), 0, _EXPA[_LOGA[a] + _LOGA[b]])


def _inv(a: int) -> int:
    return _EXP[4095 - _LOG[a]]


def _mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


CAUCHY = np.array([[_inv(j ^ (RS_P + i)) for i in range(RS_K)] for j in range(RS_P)], np.int32)   # (6, 42)


def group_sizes(n_data: int) -> list:
    return [min(RS_K, n_data - s) for s in range(0, n_data, RS_K)]


def total_strands(n_data: int, rs: bool) -> int:
    return n_data + (RS_P * len(group_sizes(n_data)) if rs else 0)


def rs_encode(letters: np.ndarray) -> np.ndarray:
    """data letters (n_data*120) -> data and parity strands interleaved per group."""
    D = np.asarray(letters, np.int32).reshape(-1, LPS)
    out = []
    for s in range(0, len(D), RS_K):
        blk = D[s:s + RS_K]
        par = np.bitwise_xor.reduce(gf_mul(CAUCHY[:, :len(blk), None], blk[None]), axis=1)
        out += [blk, par]
    return np.concatenate(out).reshape(-1)


def rs_decode(letters: np.ndarray, failed: np.ndarray, n_data: int):
    """Rebuild unreadable data letters from the readable ones. Returns (data letters, still-failed mask, number rebuilt)."""
    L = np.asarray(letters, np.int32).reshape(-1, LPS).copy()
    F = np.asarray(failed).reshape(-1, LPS).copy()
    data, left, pos, before = [], [], 0, 0
    for k in group_sizes(n_data):
        blk, fm = L[pos:pos + k + RS_P], F[pos:pos + k + RS_P]
        before += int(fm[:k].sum())
        for p in np.flatnonzero(fm[:k].any(axis=0)):
            er = np.flatnonzero(fm[:, p])
            if len(er) > RS_P:
                continue
            U = [int(e) for e in er if e < k]
            S = [j for j in range(RS_P) if (k + j) not in er][:len(U)]
            known = [i for i in range(k) if i not in U]
            rhs = blk[[k + j for j in S], p].copy()
            if known:
                rhs ^= np.bitwise_xor.reduce(gf_mul(CAUCHY[S][:, known], blk[known, p][None]), axis=1)
            A = [[int(CAUCHY[j, u]) for u in U] + [int(r)] for j, r in zip(S, rhs)]
            for c in range(len(U)):                       # Gauss-Jordan in GF(2^12)
                piv = next(r for r in range(c, len(U)) if A[r][c])
                A[c], A[piv] = A[piv], A[c]
                iv = _inv(A[c][c]); A[c] = [_mul(iv, v) for v in A[c]]
                for r in range(len(U)):
                    if r != c and A[r][c]:
                        f = A[r][c]; A[r] = [v ^ _mul(f, w) for v, w in zip(A[r], A[c])]
            for c, u in enumerate(U):
                blk[u, p], fm[u, p] = A[c][-1], False
        data.append(blk[:k]); left.append(fm[:k]); pos += k + RS_P
    after = np.concatenate(left).reshape(-1)
    return np.concatenate(data).reshape(-1), after, before - int(after.sum())


def expected_lost(copies: int, n_data: int, rs: bool) -> float:
    """Expected number of unreadable letters (no RS) or unrecoverable letter positions (RS), from sampling alone."""
    from math import comb
    q = miss_probability(copies)
    if not rs:
        return q * n_data * LPS
    tot = 0.0
    for k in group_sizes(n_data):
        n = k + RS_P
        tot += LPS * (1 - sum(comb(n, x) * q ** x * (1 - q) ** (n - x) for x in range(RS_P + 1)))
    return tot


# ----------------------------------------------------------------------------- decoder
_EXACT = np.full(65, -1, np.int16)      # trimer code (base-4, 64 = unreadable) -> index in Omega, exact match only
for _c in range(64):
    _t = "".join("ACGT"[(_c >> sh) & 3] for sh in (4, 2, 0))
    if _t in OMEGA:
        _EXACT[_c] = OMEGA.index(_t)
_LUT = np.full(256, 4, np.uint8)
for _i, _ch in enumerate(b"ACGT"):
    _LUT[_ch] = _i


def decode_reads(reads: np.ndarray, n_bytes: int | None, correct: bool, rs: bool = False, chunk: int = 1500):
    """Group molecules by barcode, vote the trimers seen at each position, rebuild the letters.
    correct=False: a position is valid only if exactly 5 Omega trimers were seen and nothing else.
    correct=True : barcode parity checked (bad barcodes dropped); reads whose trimer is not in Omega are ignored (d = 2 lets a
                   single substitution always be detected); the 5 most-voted trimers are the letter."""
    m = len(reads)
    v = _LUT[reads]
    dig = v[:, :BARCODE_LEN].astype(np.int64)
    bc_ok = (dig < 4).all(axis=1)
    if correct:                                   # parity nucleotide: a single substitution in the barcode is detected
        bc_ok &= (dig[:, :-1].sum(axis=1) % 4) == dig[:, -1]
    bc = (dig[:, :-1] << (2 * np.arange(BARCODE_LEN - 2, -1, -1))).sum(axis=1)
    bc[~bc_ok] = -1
    st = {"reads_in": m, "bad_barcode": 0, "ignored_trimers": 0}
    n_data = None
    if n_bytes is not None:
        n_data = n_strands_for(n_bytes)
        n_strands = total_strands(n_data, rs)
    else:
        uniq, cnt = np.unique(bc[bc >= 0], return_counts=True)
        if len(cnt) == 0:
            raise ValueError("No readable molecules.")
        n_strands = int(uniq[cnt >= max(2, 0.5 * np.median(cnt))].max()) + 1
    keep = (bc >= 0) & (bc < n_strands)
    st["bad_barcode"] = int(m - keep.sum())

    t = v[keep][:, BARCODE_LEN:].reshape(-1, LPS, 3).astype(np.int16)
    code = np.where((t > 3).any(axis=2), 64, t[..., 0] * 16 + t[..., 1] * 4 + t[..., 2])
    s = bc[keep]
    order = np.argsort(s, kind="stable")
    s, code = s[order], code[order]
    exact = _EXACT[code]
    vote = exact
    st["ignored_trimers"] = int((exact < 0).sum())

    posn = np.arange(LPS)[None, :]

    def vote_pass(sel):
        """Vote the trimers of the selected reads and build every letter (returns letters, failed, 16-bit trimer sets)."""
        letters = np.zeros(n_strands * LPS, np.int32)
        failed = np.ones(n_strands * LPS, bool)
        sets = np.full(n_strands * LPS, 0xFFFF, np.int32)
        ss, vv_all = s[sel], vote[sel]
        for a in range(0, n_strands, chunk):
            nb = min(chunk, n_strands - a)
            lo, hi = np.searchsorted(ss, a), np.searchsorted(ss, a + nb)
            base = (ss[lo:hi, None] - a) * LPS + posn
            vv = vv_all[lo:hi].astype(np.int64)
            good = vv >= 0
            votes = np.bincount((base * 16 + vv)[good], minlength=nb * LPS * 16).reshape(nb, LPS, 16)
            bad = np.bincount(base[~good], minlength=nb * LPS).reshape(nb, LPS)
            npres = (votes > 0).sum(axis=2)
            order = np.argsort(-votes, axis=2, kind="stable")
            top5 = order[..., :K]
            mask = (1 << top5).sum(axis=2)
            ok = (npres >= K) if correct else ((npres == K) & (bad == 0))
            if correct:        # 5th and 6th trimer equally supported -> cannot tell which is real: call it unreadable
                v56 = np.take_along_axis(votes, order[..., K - 1:K + 1], axis=2)
                ok &= ~((v56[..., 0] == v56[..., 1]) & (v56[..., 1] > 0))
            let = MASK_TO_LETTER[mask]
            ok &= let >= 0
            letters[a * LPS:(a + nb) * LPS] = np.where(ok, let, 0).ravel()
            failed[a * LPS:(a + nb) * LPS] = (~ok).ravel()
            sets[a * LPS:(a + nb) * LPS] = np.where(ok, mask, 0xFFFF).ravel()
        return letters, failed, sets

    letters, failed, sets = vote_pass(np.ones(len(s), bool))
    st["misplaced_reads"] = 0
    if correct:
        # A read that was filed under the wrong strand (barcode hit by 2 substitutions) disagrees with the consensus
        # of that strand at most positions. Throw such reads away and vote again.
        member = (sets.reshape(n_strands, LPS)[s] >> np.maximum(vote, 0)) & 1
        seen = vote >= 0
        agree = (member * seen).sum(axis=1) / np.maximum(seen.sum(axis=1), 1)
        sel = agree >= 0.6
        st["misplaced_reads"] = int((~sel).sum())
        if st["misplaced_reads"]:
            letters, failed, sets = vote_pass(sel)
    st["unreadable_letters"] = int(failed.sum())
    st["letters"] = len(letters)
    st["strands"] = n_strands
    st["rs_recovered"] = 0
    if rs and n_data is not None:
        letters, failed, st["rs_recovered"] = rs_decode(letters, failed, n_data)
    st["still_unreadable"] = int(failed.sum())
    st["strands_missing"] = int(n_strands - len(np.unique(s)))
    return letters, st
