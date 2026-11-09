import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import shortmer_core as C

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
    out, _ = C.decode_reads(r2, meta["bytes"], True, False)
    assert C.letters_to_bytes(out, len(d)) == d


def test_gf_primitive():
    assert len(set(C._EXP[:4095])) == 4095

def test_rs_recovers_erasures():
    rng = np.random.default_rng(0)
    for n_data in (1, 5, 42, 100):
        L = rng.integers(0, 4096, n_data * 120)
        full = C.rs_encode(L)
        assert len(full) == C.total_strands(n_data, True) * 120
        failed = np.zeros(len(full), bool)
        M = full.reshape(-1, 120).copy(); F = failed.reshape(-1, 120)
        pos = 0
        for k in C.group_sizes(n_data):                    # erase 6 random strands' letters in every column
            for p in range(120):
                for r in rng.choice(k + 6, size=min(6, k + 6), replace=False):
                    F[pos + r, p] = True; M[pos + r, p] = 0
            pos += k + 6
        out, left, rec = C.rs_decode(M.reshape(-1), F.reshape(-1), n_data)
        assert (out == L).all() and not left.any()

def test_end_to_end_with_rs_few_copies():
    d = os.urandom(90000); Ld = C.bytes_to_letters(d); L = C.rs_encode(Ld)
    clean, _ = C.make_reads(L, 25, 0.0)
    reads, _ = C.add_substitutions(clean, 0.001)
    out, st = C.decode_reads(reads, len(d), True, True)
    assert C.letters_to_bytes(out, len(d)) == d and st["unreadable_letters"] > 0 and st["still_unreadable"] == 0
