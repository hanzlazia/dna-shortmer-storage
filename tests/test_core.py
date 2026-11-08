import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import dna_codec as C

def test_omega_distance_2():
    assert min(sum(a != b for a, b in zip(x, y)) for i, x in enumerate(C.OMEGA) for y in C.OMEGA[i + 1:]) == 2

def test_roundtrip_bits():
    d = os.urandom(5000)
    assert C.letters_to_bytes(C.bytes_to_letters(d), len(d)) == d

def _run(d, copies, p, correct):
    L = C.bytes_to_letters(d)
    reads, _ = C.make_reads(L, copies, p)
    out, st = C.decode_reads(reads, len(d), correct)
    return C.letters_to_bytes(out, len(d)) == d, st

def test_clean_exact():
    d = os.urandom(20000); assert _run(d, 60, 0.0, True)[0] and _run(d, 60, 0.0, False)[0]

def test_correction_recovers_substitutions():
    d = os.urandom(20000)
    L = C.bytes_to_letters(d); reads, _ = C.make_reads(L, 60, 0.005)
    good = (C.decode_reads(reads, len(d), True)[0] != L).sum()
    plain = (C.decode_reads(reads, len(d), False)[0] != L).sum()
    assert good < 0.001 * len(L) and plain > 20 * max(good, 1)

def test_fastq_roundtrip():
    d = os.urandom(3000); L = C.bytes_to_letters(d)
    reads, _ = C.make_reads(L, 120, 0.0)
    fq = C.to_fastq(reads, 120, dict(name="a b.jpg", bytes=len(d), sha="abc123", comp="lzma"))
    r2, meta, bad = C.parse_fastq(fq)
    assert meta["name"] == "a b.jpg" and meta["comp"] == "lzma" and bad == 0 and (r2 == reads).all()
    out, _ = C.decode_reads(r2, meta["bytes"], True)
    assert C.letters_to_bytes(out, len(d)) == d
