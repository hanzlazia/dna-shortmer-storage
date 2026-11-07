"""Storage statistics and comparison with the simple 2-bit-per-nucleotide mapping."""
from __future__ import annotations

from collections import Counter
from math import ceil

from .codec import Encoded

_MAP = ["".join("ACGT"[(b >> s) & 3] for s in (6, 4, 2, 0)) for b in range(256)]


def simple_dna(data: bytes) -> str:
    """Standard mapping 00->A, 01->C, 10->G, 11->T, MSB first (no error correction)."""
    return "".join(_MAP[b] for b in data)


def max_homopolymer(s: str) -> int:
    best = cur = 0
    prev = ""
    for ch in s:
        cur = cur + 1 if ch == prev else 1
        prev = ch
        best = max(best, cur)
    return best


def composition(s: str) -> dict[str, int]:
    c = Counter(s)
    return {nt: c.get(nt, 0) for nt in "ACGT"}


def gc(counts: dict) -> float:
    tot = sum(counts.values())
    return 100 * (counts["G"] + counts["C"]) / tot if tot else 0.0


def encoding_stats(enc: Encoded, data: bytes) -> dict:
    p, a = enc.params, enc.alphabet
    r, l, W, rows = a.bits, p.payload_letters, p.width, enc.total_rows
    nbits = len(data) * 8

    # --- simple mapping ----------------------------------------------------
    std = simple_dna(data)
    bc_nt = (enc.nb + p.bc_parity) * 2
    mol_nt = bc_nt + W * a.k
    simple = {
        "nt": len(std), "counts": composition(std), "max_homopolymer": max_homopolymer(std),
        "oligos_same_length": ceil(len(std) / mol_nt), "preview_bits": enc.bits[:48],
        "preview_nt": std[:24],
    }

    # --- combinatorial design ----------------------------------------------
    bc_counts = composition("".join(enc.barcodes))
    cache: dict[int, dict[str, float]] = {}
    exp = {nt: 0.0 for nt in "ACGT"}
    for row in enc.matrix:
        for v in row:
            if v not in cache:
                c = {nt: 0.0 for nt in "ACGT"}
                for km in a.kmers(v):
                    for ch in km:
                        c[ch] += 1 / a.K
                cache[v] = c
            for nt in "ACGT":
                exp[nt] += cache[v][nt]
    total_counts = {nt: bc_counts[nt] + exp[nt] for nt in "ACGT"}
    cycles = rows * (bc_nt + W)
    comb = {
        "sequences": rows, "molecule_nt": mol_nt, "total_nt": rows * mol_nt,
        "cycles": cycles, "kmer_injections": rows * W * a.K,
        "barcode_counts": bc_counts, "payload_expected": exp, "total_expected": total_counts,
    }

    # --- where do the stored bits go ----------------------------------------
    data_rows = enc.n_blocks * p.block_data
    parts = {
        "Message": nbits,
        "Padding (zero bits / letters)": enc.pad_bits_total,
        "Padding-information sequence": l * r,
        "Lengthwise RS": data_rows * p.rs_len * r,
        "Crosswise RS": enc.n_blocks * p.block_parity * W * r,
    }
    parts_total = sum(parts.values())
    assert parts_total == rows * W * r, (parts_total, rows * W * r)
    return {
        "bits": nbits, "bytes": len(data), "bits_per_letter": r, "bit_pad": enc.bit_pad,
        "letters": enc.n_data_letters, "simple": simple, "comb": comb, "parts": parts,
        "parts_total": parts_total, "barcode_nt_total": rows * bc_nt,
        "info_bits_per_cycle": nbits / cycles, "fold_vs_simple": (len(std)) / cycles,
    }


def report_text(enc: Encoded, data: bytes, name: str) -> str:
    s = encoding_stats(enc, data)
    p, a = enc.params, enc.alphabet
    sm, cb = s["simple"], s["comb"]
    L = [f"Combinatorial shortmer DNA storage - encoding report",
         f"File: {name}", f"Size: {s['bytes']:,} bytes = {s['bits']:,} bits", "",
         "Scheme",
         f"  Omega: N={a.N} {a.k}-mers, min Hamming distance {a.min_distance}",
         f"  Letter = K={a.K} of N={a.N} k-mers; |Sigma|={a.size:,}; {a.bits} bits per letter",
         f"  Payload letters/sequence={p.payload_letters}; lengthwise RS parity={p.rs_len}; "
         f"block={p.block_data}+{p.block_parity}; barcode RS parity={p.bc_parity}", "",
         "Simple mapping (2 bits per nucleotide, no error correction)",
         f"  nucleotides: {sm['nt']:,}  A={sm['counts']['A']:,} C={sm['counts']['C']:,} "
         f"G={sm['counts']['G']:,} T={sm['counts']['T']:,}  GC={gc(sm['counts']):.1f}%",
         f"  longest homopolymer: {sm['max_homopolymer']}",
         f"  synthesis cycles: {sm['nt']:,}  (2.00 bits/cycle)", "",
         "Combinatorial shortmer encoding (all overhead included)",
         f"  sequences: {cb['sequences']:,}; molecule length {cb['molecule_nt']} nt; total {cb['total_nt']:,} nt",
         f"  synthesis cycles: {cb['cycles']:,}  ({s['info_bits_per_cycle']:.2f} information bits/cycle)",
         f"  k-mer injections: {cb['kmer_injections']:,}",
         f"  expected composition: A={cb['total_expected']['A']:,.1f} C={cb['total_expected']['C']:,.1f} "
         f"G={cb['total_expected']['G']:,.1f} T={cb['total_expected']['T']:,.1f}  "
         f"GC={gc(cb['total_expected']):.1f}%",
         f"  cycles relative to simple mapping: {cb['cycles'] / sm['nt']:.3f}x "
         f"({s['fold_vs_simple']:.2f}x fewer)", "",
         "Stored bits breakdown"]
    for k, v in s["parts"].items():
        L.append(f"  {k}: {v:,} bits ({100 * v / s['parts_total']:.1f}%)")
    L.append(f"  Total: {s['parts_total']:,} bits")
    return "\n".join(L) + "\n"
