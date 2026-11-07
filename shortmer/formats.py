"""FASTA / FASTQ export of designed molecules and simulated sequencing reads.

A combinatorial letter is a *mixture* of k-mers, so a single sequence has no
single DNA string. Exports therefore contain concrete molecules drawn from the
mixture, exactly as the synthesis would produce them.
"""
from __future__ import annotations

import numpy as np

from .codec import Encoded

NT = "ACGT"


def sample_molecule(enc: Encoded, i: int, rng: np.random.Generator) -> str:
    """One concrete molecule of sequence i: barcode + one random k-mer per letter."""
    a = enc.alphabet
    parts = [enc.barcodes[i]]
    for v in enc.matrix[i]:
        ks = a.kmers(v)
        parts.append(ks[int(rng.integers(0, len(ks)))])
    return "".join(parts)


def design_records(enc: Encoded, rng: np.random.Generator, copies: int = 1):
    out = []
    for i in range(enc.total_rows):
        for c in range(copies):
            out.append((f"seq{i:05d}_mol{c + 1:03d} role={enc.row_kind[i]} barcode={enc.barcodes[i]}",
                        sample_molecule(enc, i, rng)))
    return out


def read_to_nt(barcode_digits, payload_codes, k: int) -> str:
    bc = "".join(NT[int(d)] for d in barcode_digits)
    pay = []
    for code in payload_codes:
        code = int(code)
        pay.append("".join(NT[(code // 4 ** (k - 1 - j)) % 4] for j in range(k)))
    return bc + "".join(pay)


def to_fasta(records, width: int = 80) -> str:
    lines = []
    for name, seq in records:
        lines.append(f">{name}")
        lines.extend(seq[i:i + width] for i in range(0, len(seq), width))
    return "\n".join(lines) + "\n"


def to_fastq(records, quality: str = "I") -> str:
    """records: (name, seq). Constant Phred quality (default 'I' = Q40)."""
    return "".join(f"@{name}\n{seq}\n+\n{quality * len(seq)}\n" for name, seq in records)
