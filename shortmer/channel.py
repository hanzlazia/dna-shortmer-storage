"""Error channels between encoding and decoding.

1. ``inject_letter_errors`` - transparent, symbol-level fault injection on the
   encoded letter matrix (for explaining what each RS layer can repair).
2. ``simulate_reads`` / ``reconstruct`` - molecular model following Sec. 3.1 of
   the paper: every molecule picks one random k-mer of each letter's subset,
   substitution errors act on nucleotides, insertions/deletions act on whole
   k-mers; reads are grouped by (RS-protected) barcode and every letter is
   re-inferred as the K most frequent k-mers of its column.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alphabet import Alphabet
from .codec import ERASED, Encoded, Params, barcode_decode

# ---------------------------------------------------------------------------
# 1. direct injection
# ---------------------------------------------------------------------------


@dataclass
class Injection:
    received: list[list[int] | None]
    touched: set[tuple[int, int]] = field(default_factory=set)
    lost_rows: list[int] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


def inject_letter_errors(enc: Encoded, rng: np.random.Generator, *,
                         random_errors: int = 0, burst_row: int | None = None,
                         burst_len: int = 0, lose_rows: int = 0,
                         targeted: list[tuple[int, int, int]] = ()) -> Injection:
    """Corrupt the letter matrix.

    random_errors : substitutions at uniformly random cells
    burst_row/len : `burst_len` substitutions inside one sequence
    lose_rows     : number of whole sequences dropped (synthesis/sequencing dropout)
    targeted      : explicit (row, col, new_letter) edits
    """
    rows = [list(r) for r in enc.matrix]
    W, T, usable = enc.params.width, len(rows), enc.alphabet.usable
    inj = Injection(received=rows)

    def corrupt(i: int, c: int, new: int | None = None):
        old = rows[i][c]
        if new is None or new == old:
            new = int(rng.integers(0, usable - 1))
            if new >= old:
                new += 1
        rows[i][c] = new
        inj.touched.add((i, c))
        inj.log.append(f"sequence {i}, position {c}: letter {old} -> {new}")

    for i, c, v in targeted:
        if 0 <= i < T and 0 <= c < W:
            corrupt(i, c, v)
    for flat in rng.choice(T * W, size=min(random_errors, T * W), replace=False):
        corrupt(int(flat) // W, int(flat) % W)
    if burst_row is not None and burst_len > 0 and 0 <= burst_row < T:
        for c in rng.choice(W, size=min(burst_len, W), replace=False):
            corrupt(burst_row, int(c))
    if lose_rows:
        gone = rng.choice(T, size=min(lose_rows, T), replace=False)
        for i in gone:
            rows[int(i)] = None
            inj.lost_rows.append(int(i))
            inj.log.append(f"sequence {int(i)}: lost completely (dropout)")
    inj.lost_rows.sort()
    return inj


# ---------------------------------------------------------------------------
# 2. molecular channel
# ---------------------------------------------------------------------------


@dataclass
class ChannelStats:
    reads: int = 0
    discarded_barcode: int = 0
    sub_events: int = 0
    del_events: int = 0
    ins_events: int = 0
    groups: int = 0
    mean_depth: float = 0.0


def _kmer_code_table(alpha: Alphabet) -> tuple[np.ndarray, np.ndarray]:
    """(codes of Omega k-mers, lookup code -> Omega index or -1)."""
    k = alpha.k
    codes = np.array([sum("ACGT".index(ch) * 4 ** (k - 1 - i)
                          for i, ch in enumerate(w)) for w in alpha.omega])
    lut = np.full(4 ** k, -1, dtype=np.int64)
    lut[codes] = np.arange(len(codes))
    return codes, lut


def simulate_reads(enc: Encoded, rng: np.random.Generator, *, depth: float,
                   p_sub: float, p_ins: float, p_del: float,
                   lose_rows: int = 0):
    """Return (reads, stats). A read is (barcode_digits ndarray, payload_codes ndarray)."""
    alpha, p = enc.alphabet, enc.params
    k, W, K = alpha.k, p.width, alpha.K
    codes, _ = _kmer_code_table(alpha)
    bc_digits = np.array([[("ACGT".index(ch)) for ch in bc] for bc in enc.barcodes])
    nbc = bc_digits.shape[1]
    subsets = np.array([alpha.subset(v) for v in _unique_letters(enc)])  # (U, K)
    letter_to_u = {v: i for i, v in enumerate(_unique_letters(enc))}

    gone = set(rng.choice(enc.total_rows, size=min(lose_rows, enc.total_rows),
                          replace=False).tolist()) if lose_rows else set()
    st = ChannelStats()
    reads = []
    for i in range(enc.total_rows):
        if i in gone:
            continue
        n = int(rng.poisson(depth))
        if n == 0:
            continue
        u = np.array([letter_to_u[v] for v in enc.matrix[i]])
        pick = rng.integers(0, K, size=(n, W))
        kid = subsets[u[None, :], pick]                       # (n, W) Omega indices
        pay = codes[kid]                                      # k-mer codes
        # nucleotide substitutions (inside k-mers and barcode)
        if p_sub > 0:
            digs = np.stack([(pay // 4 ** (k - 1 - j)) % 4 for j in range(k)], -1)
            m = rng.random(digs.shape) < p_sub
            st.sub_events += int(m.sum())
            digs = np.where(m, (digs + rng.integers(1, 4, size=digs.shape)) % 4, digs)
            pay = sum(digs[..., j] * 4 ** (k - 1 - j) for j in range(k))
            bd = np.broadcast_to(bc_digits[i], (n, nbc))
            mb = rng.random(bd.shape) < p_sub
            bd = np.where(mb, (bd + rng.integers(1, 4, size=bd.shape)) % 4, bd)
        else:
            bd = np.broadcast_to(bc_digits[i], (n, nbc))
        for j in range(n):
            # barcode indels -> misaligned barcode, read cannot be assigned
            if (p_ins or p_del) and rng.random() < 1 - (1 - p_ins - p_del) ** nbc:
                st.discarded_barcode += 1
                st.reads += 1
                continue
            row = pay[j]
            if p_del or p_ins:
                dm = rng.random(W) < p_del
                im = rng.random(W) < p_ins
                if dm.any() or im.any():
                    out = []
                    for pos in range(W):
                        if im[pos]:
                            out.append(int(rng.integers(0, 4 ** k)))
                            st.ins_events += 1
                        if dm[pos]:
                            st.del_events += 1
                        else:
                            out.append(int(row[pos]))
                    row = np.array(out)
            reads.append((bd[j].copy(), np.asarray(row)))
            st.reads += 1
    return reads, st


def _unique_letters(enc: Encoded) -> list[int]:
    return sorted({v for row in enc.matrix for v in row})


def reconstruct(reads, p: Params, alpha: Alphabet, stats: ChannelStats | None = None,
                total_rows: int | None = None):
    """Group reads by barcode and infer every combinatorial letter.

    Returns (matrix with None rows for unseen barcodes / -1 for unresolved
    letters, ChannelStats).
    """
    stats = stats or ChannelStats()
    W, K, N = p.width, alpha.K, alpha.N
    _, lut = _kmer_code_table(alpha)
    groups: dict[int, list[np.ndarray]] = {}
    for bd, pay in reads:
        syms = [(int(bd[i]) << 2) | int(bd[i + 1]) for i in range(0, len(bd), 2)]
        idx = barcode_decode(syms, p.bc_parity)
        if idx is None:
            stats.discarded_barcode += 1
            continue
        groups.setdefault(idx, []).append(pay)
    if not groups:
        return [], stats
    n_rows = total_rows or (-(-(max(groups) + 1) // p.block_size) * p.block_size)
    out: list[list[int] | None] = [None] * n_rows
    stats.groups = len(groups)
    stats.mean_depth = float(np.mean([len(v) for v in groups.values()]))
    for idx, pays in groups.items():
        if idx >= n_rows:
            continue
        mat = np.full((len(pays), W), -1, dtype=np.int64)
        for j, pay in enumerate(pays):          # paper 3.2: pad short / cut long reads
            m = min(W, len(pay))
            mat[j, :m] = pay[:m]
        om = np.where(mat >= 0, lut[np.clip(mat, 0, len(lut) - 1)], -1)
        row = []
        for c in range(W):
            col = om[:, c]
            cnt = np.bincount(col[col >= 0], minlength=N)
            top = np.argsort(-cnt, kind="stable")[:K]
            if cnt[top[-1]] == 0:
                row.append(ERASED)               # fewer than K distinct valid k-mers
                continue
            letter = alpha.letter(tuple(int(t) for t in top))
            row.append(ERASED if letter is None else letter)
        out[idx] = row
    return out, stats
