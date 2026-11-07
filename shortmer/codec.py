"""End-to-end combinatorial shortmer codec (paper Sec. 2.1, 3.1, 3.2).

Encoding pipeline
    bytes -> bits -> combinatorial letters (r bits / letter)
          -> sequences of `payload_letters` letters
          -> + lengthwise RS over GF(2^r) on every sequence
          -> blocks of `block_data` sequences, + crosswise RS (columns) giving
             `block_parity` extra sequences per block
          -> barcodes (index protected by RS over GF(2^4)), 2 nt per symbol

Decoding mirrors the steps in reverse and can use erasures (missing sequences,
unresolved positions) in addition to unknown-location errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .alphabet import NT, Alphabet, build_omega
from .rs import RSCodec

# cell categories used for the 2D layout figure
PAYLOAD, PADDING, PADINFO, LRS, XRS = (
    "Payload", "Padding", "Padding info", "Lengthwise RS", "Crosswise RS")
KIND_ORDER = [PAYLOAD, LRS, XRS, PADDING, PADINFO]


@dataclass(frozen=True)
class Params:
    N: int = 16                 # |Omega|, number of k-mers
    K: int = 5                  # k-mers per letter (binomial encoding)
    k: int = 3                  # k-mer length (trimers)
    d: int = 2                  # minimum Hamming distance of Omega
    payload_letters: int = 40   # data letters per sequence (l)
    rs_len: int = 6             # lengthwise RS parity letters per sequence
    block_data: int = 10        # data sequences per block
    block_parity: int = 4       # crosswise parity sequences per block
    bc_parity: int = 2          # RS parity symbols (GF(16), 2 nt each) on barcode

    @property
    def width(self) -> int:
        return self.payload_letters + self.rs_len

    @property
    def block_size(self) -> int:
        return self.block_data + self.block_parity


PRESETS = {
    "Demo (small, fast)": Params(),
    "Paper, Sec. 3.1 (N=16, K=5)": Params(
        N=16, K=5, k=3, d=2, payload_letters=120, rs_len=14,
        block_data=42, block_parity=6, bc_parity=2),
    "Paper, Fig. 6 (N=16, K=3)": Params(
        N=16, K=3, k=3, d=2, payload_letters=12, rs_len=6,
        block_data=10, block_parity=5, bc_parity=2),
}


def make_alphabet(p: Params) -> Alphabet:
    return Alphabet(build_omega(p.k, p.N, p.d), p.K)


# --------------------------------------------------------------------------
# bit / letter helpers
# --------------------------------------------------------------------------
def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def bits_to_bytes(bits: str) -> bytes:
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _int_to_letters(value: int, n: int, bits: int) -> list[int]:
    out = []
    for _ in range(n):
        out.append(value & ((1 << bits) - 1))
        value >>= bits
    if value:
        raise ValueError("value does not fit in the padding-info sequence")
    return out


def _letters_to_int(letters: list[int], bits: int) -> int:
    v = 0
    for i, x in enumerate(letters):
        v |= x << (bits * i)
    return v


# --------------------------------------------------------------------------
# barcodes
# --------------------------------------------------------------------------
def barcode_data_symbols(total_rows: int) -> int:
    """Number of GF(16) symbols needed to index `total_rows` sequences."""
    n, cap = 1, 16
    while cap < total_rows:
        n += 1
        cap *= 16
    return n


def barcode_encode(index: int, nb: int, bc_parity: int) -> list[int]:
    syms = [(index >> (4 * (nb - 1 - i))) & 15 for i in range(nb)]
    return RSCodec(4, bc_parity).encode(syms)


def barcode_decode(symbols: list[int], bc_parity: int) -> int | None:
    fixed, _ = RSCodec(4, bc_parity).decode(symbols)
    if fixed is None:
        return None
    v = 0
    for s in fixed[: len(fixed) - bc_parity]:
        v = (v << 4) | s
    return v


def symbols_to_nt(symbols: list[int]) -> str:
    return "".join(NT[(s >> 2) & 3] + NT[s & 3] for s in symbols)


def nt_to_symbols(nts: str) -> list[int]:
    idx = [NT.index(c) for c in nts]
    return [(idx[i] << 2) | idx[i + 1] for i in range(0, len(idx), 2)]


# --------------------------------------------------------------------------
# result containers
# --------------------------------------------------------------------------
@dataclass
class Encoded:
    params: Params
    alphabet: Alphabet
    n_bytes: int
    bits: str
    bit_pad: int                      # zero bits appended to fill the last letter
    data_letters: list[int]           # letters carrying the message (+bit pad)
    n_data_letters: int               # == len(data_letters)
    pad_bits_total: int               # all padding (bit + letter + sequence)
    n_blocks: int
    matrix: list[list[int]]           # total_rows x width, letters
    row_kind: list[str]               # 'data' | 'info' | 'xrs' per row
    kinds: list[list[str]]            # category of every cell
    nb: int                           # barcode data symbols
    barcode_symbols: list[list[int]]
    barcodes: list[str]               # barcode as nucleotides

    @property
    def total_rows(self) -> int:
        return len(self.matrix)

    @property
    def sequences(self) -> list[tuple[str, list[int]]]:
        return list(zip(self.barcodes, self.matrix))


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def encode(data: bytes, p: Params) -> Encoded:
    if not data:
        raise ValueError("empty input")
    alpha = make_alphabet(p)
    r = alpha.bits
    l, W = p.payload_letters, p.width
    if W > (1 << r) - 1:
        raise ValueError(f"sequence length {W} exceeds RS limit 2^{r}-1")
    if p.block_size > (1 << r) - 1:
        raise ValueError("block too large for the field")
    if p.bc_parity < 2:
        raise ValueError("barcode needs >= 2 parity symbols")

    bits = bytes_to_bits(data)
    bit_pad = (-len(bits)) % r
    padded = bits + "0" * bit_pad
    letters = [int(padded[i:i + r], 2) for i in range(0, len(padded), r)]
    Ld = len(letters)

    n_seq = -(-Ld // l)                       # data sequences holding the message
    rows_needed = n_seq + 1                   # + padding-info sequence
    n_blocks = -(-rows_needed // p.block_data)
    data_rows = n_blocks * p.block_data
    # all padding: appended zero bits + zero letters up to the info sequence
    pad_bits_total = (data_rows - 1) * l * r - len(bits)

    flat = letters + [0] * ((data_rows - 1) * l - Ld)
    rows = [flat[i * l:(i + 1) * l] for i in range(data_rows - 1)]
    info = _int_to_letters(pad_bits_total, l, r)
    rows.append(info)

    # lengthwise RS (every data sequence)
    lrs = RSCodec(r, p.rs_len)
    rows = [lrs.encode(row) for row in rows]

    # crosswise RS (columns of each block), then assemble in block order
    xrs = RSCodec(r, p.block_parity)
    matrix: list[list[int]] = []
    row_kind: list[str] = []
    kinds: list[list[str]] = []
    for b in range(n_blocks):
        blk = rows[b * p.block_data:(b + 1) * p.block_data]
        cols = [xrs.parity([blk[i][c] for i in range(p.block_data)])
                for c in range(W)]
        par_rows = [[cols[c][j] for c in range(W)] for j in range(p.block_parity)]
        for i, row in enumerate(blk):
            g = b * p.block_data + i              # global data-row index
            matrix.append(row)
            if g == data_rows - 1:
                row_kind.append("info")
                kinds.append([PADINFO] * W)
            else:
                row_kind.append("data")
                kinds.append([
                    (PAYLOAD if g * l + c < Ld else PADDING) if c < l else LRS
                    for c in range(W)])
        for row in par_rows:
            matrix.append(row)
            row_kind.append("xrs")
            kinds.append([XRS] * W)

    total = len(matrix)
    nb = barcode_data_symbols(total)
    if nb + p.bc_parity > 15:
        raise ValueError("too many sequences for a GF(16) barcode")
    bc_syms = [barcode_encode(i, nb, p.bc_parity) for i in range(total)]
    return Encoded(
        params=p, alphabet=alpha, n_bytes=len(data), bits=bits, bit_pad=bit_pad,
        data_letters=letters, n_data_letters=Ld, pad_bits_total=pad_bits_total,
        n_blocks=n_blocks, matrix=matrix, row_kind=row_kind, kinds=kinds, nb=nb,
        barcode_symbols=bc_syms, barcodes=[symbols_to_nt(s) for s in bc_syms])


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
ERASED = -1


@dataclass
class DecodeReport:
    recovered: bytes | None
    ok: bool
    message: str
    stage_matrices: dict[str, list[list[int]]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)


def decode(received: list[list[int] | None], p: Params,
           alpha: Alphabet | None = None) -> DecodeReport:
    """Decode a (possibly damaged) letter matrix.

    `received[i]` is the i-th sequence (None if lost); letters equal to
    ``ERASED`` (-1) are unresolved positions (known erasures).
    """
    alpha = alpha or make_alphabet(p)
    r, W, l = alpha.bits, p.width, p.payload_letters
    total = len(received)
    stats = dict(rows_missing=sum(x is None for x in received),
                 lengthwise_failed=0, lengthwise_symbols_fixed=0,
                 crosswise_failed_columns=0, crosswise_symbols_fixed=0)
    stages = {"received": [list(x) if x else [ERASED] * W for x in received]}

    # --- lengthwise RS on each sequence -----------------------------------
    lrs = RSCodec(r, p.rs_len)
    stage1: list[list[int]] = []
    row_bad: list[bool] = []
    for row in received:
        if row is None:
            stage1.append([ERASED] * W)
            row_bad.append(True)
            continue
        er = [i for i, v in enumerate(row) if v == ERASED]
        fixed, nfix = lrs.decode([max(v, 0) for v in row], er)
        if fixed is None:
            stats["lengthwise_failed"] += 1
            stage1.append([ERASED if v == ERASED else v for v in row])
            row_bad.append(True)
        else:
            stats["lengthwise_symbols_fixed"] += nfix
            stage1.append(fixed)
            row_bad.append(False)
    stages["after lengthwise RS"] = [list(x) for x in stage1]

    # --- crosswise RS on every column of each block ------------------------
    xrs = RSCodec(r, p.block_parity)
    bs = p.block_size
    n_blocks = -(-total // bs)
    stage2 = [list(x) for x in stage1]
    for b in range(n_blocks):
        idx = list(range(b * bs, min((b + 1) * bs, total)))
        if len(idx) != bs:       # truncated final block: cannot decode it
            for i in idx:
                row_bad[i] = True
            continue
        bad_rows = [j for j, i in enumerate(idx) if row_bad[i]]
        for c in range(W):
            col = [max(stage1[i][c], 0) for i in idx]
            fixed, nfix = xrs.decode(col, bad_rows)
            if fixed is None:
                stats["crosswise_failed_columns"] += 1
                continue
            stats["crosswise_symbols_fixed"] += nfix
            for j, i in enumerate(idx):
                stage2[i][c] = fixed[j]
    stages["after crosswise RS"] = stage2

    # --- reassemble message -------------------------------------------------
    data_rows: list[list[int]] = []
    for b in range(n_blocks):
        for j in range(p.block_data):
            i = b * bs + j
            if i >= total:
                return DecodeReport(None, False, "incomplete final block",
                                    stages, stats)
            data_rows.append(stage2[i][:l])
    if any(v == ERASED for row in data_rows for v in row):
        return DecodeReport(None, False,
                            "unrecoverable: erased symbols remain", stages, stats)
    info = data_rows[-1]
    try:
        pad_bits = _letters_to_int(info, r)
    except Exception:
        pad_bits = -1
    body = [v for row in data_rows[:-1] for v in row]
    nbits = len(body) * r - pad_bits
    if pad_bits < 0 or nbits <= 0 or nbits % 8:
        return DecodeReport(None, False, "padding information is inconsistent",
                            stages, stats)
    bitstr = "".join(f"{v:0{r}b}" for v in body)[:nbits]
    return DecodeReport(bits_to_bytes(bitstr), True, "decoded", stages, stats)
