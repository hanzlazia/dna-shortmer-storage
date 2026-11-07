"""Logical-density calculations (paper Sec. 4.1, Table 3)."""
from __future__ import annotations

from math import ceil, comb, floor, log2

import pandas as pd


def capacity_row(kind: str, N: int | None, K: int | None, *, payload_letters: int = 120,
                 rs_len: int = 14, barcode_nt: int = 16, block_data: int = 42,
                 block_parity: int = 6, size_bytes: float = 1e9) -> dict:
    """One row of Table 3 for `size_bytes` of data."""
    if kind == "Standard":
        alpha, bits = 4, 2
        payload = payload_letters + rs_len
        bits_seq = 2 * payload_letters
    elif kind == "Binomial":
        alpha = comb(N, K)
        bits = floor(log2(alpha))
        alpha = 2 ** bits
        bits_seq = bits * payload_letters
    else:  # Binary
        alpha = 2 ** N - 1
        bits = floor(log2(alpha))
        alpha = 2 ** bits
        bits_seq = bits * payload_letters
    payload_total = payload_letters + rs_len
    rate = block_data / (block_data + block_parity)
    n_seq = ceil(size_bytes * 8 / bits_seq)
    n_seq_rs = ceil(n_seq * (block_data + block_parity) / block_data)
    return {
        "Type": kind, "N": N, "K": K, "Bits / letter": bits, "Alphabet size": alpha,
        "Bits / sequence": bits_seq, "Sequences": n_seq, "Sequences (+ RS)": n_seq_rs,
        "Bits / cycle (payload only)": round(bits_seq / payload_total * rate, 2),
        "Bits / cycle": round(bits_seq / (payload_total + barcode_nt) * rate, 2),
        "Fold increase": round(bits / 2, 1),
    }


def table3(**kw) -> pd.DataFrame:
    specs = [("Standard", None, None), ("Binomial", 16, 3), ("Binomial", 16, 5),
             ("Binomial", 16, 7), ("Binary", 10, None), ("Binary", 16, None),
             ("Binary", 20, None)]
    return pd.DataFrame([capacity_row(*s, **kw) for s in specs])
