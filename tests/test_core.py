import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import shortmer_core as C

PLAIN = dict(barcode_check=False, ignore_garbage=False, majority=False, consensus=False)

def test_omega_distance_2():
    assert min(sum(a != b for a, b in zip(x, y)) for i, x in enumerate(C.OMEGA) for y in C.OMEGA[i + 1:]) == 2

def test_roundtrip_bits():
    d = os.urandom(5000)
    assert C.letters_to_bytes(C.bytes_to_letters(d), len(d)) == d

def test_gf_primitive():
    assert len(set(C._EXP[:4095])) == 4095

def test_rs_recovers_erasures_any_k_p():
    rng = np.random.default_rng(0)
    for (k, p), n_data in [((42, 6), 100), ((10, 3), 25), ((5, 1), 12), ((42, 6), 1), ((20, 8), 20)]:
        L = rng.integers(0, 4096, n_data * 120)
        full = C.rs_encode(L, (k, p))
        assert len(full) == C.total_strands(n_data, (k, p)) * 120
        M = full.reshape(-1, 120).copy(); F = np.zeros(M.shape, bool); pos = 0
        for g in C.group_sizes(n_data, k):
            for c in range(120):
                for r in rng.choice(g + p, size=min(p, g + p), replace=False):
                    F[pos + r, c] = True; M[pos + r, c] = 0
            pos += g + p
        out, left, rec = C.rs_decode(M.reshape(-1), F.reshape(-1), n_data, (k, p))
        assert (out == L).all() and not left.any()

def _enc(d, copies, p, rs, seed=1):
    Ld = C.bytes_to_letters(d); L = C.rs_encode(Ld, rs) if rs else Ld
    clean, _ = C.make_reads(L, copies, 0.0, seed)
    reads, n, q = C.add_substitutions(clean, p, seed)
    return Ld, reads, q

def test_fastq_roundtrip_with_quality():
    d = os.urandom(3000); Ld, reads, q = _enc(d, 120, 0.002, None)
    fq = C.to_fastq(reads, 120, dict(name="a b.jpg", bytes=len(d), sha="abc123", rsk=0), q)
    r2, q2, meta, bad = C.parse_fastq(fq)
    assert meta["name"] == "a b.jpg" and bad == 0 and (r2 == reads).all() and (q2 == q).all()

def test_plain_fails_but_full_decoder_works():
    d = os.urandom(20000); Ld, reads, q = _enc(d, 80, 0.005, None)
    good = (C.decode_reads(reads, len(d))[0] != Ld).sum()
    plain = (C.decode_reads(reads, len(d), **PLAIN)[0] != Ld).sum()
    assert good < 0.001 * len(Ld) and plain > 20 * max(good, 1)

def test_qscore_filter_helps_plain_decoder():
    d = os.urandom(20000); Ld, reads, q = _enc(d, 80, 0.005, None)
    no_q = (C.decode_reads(reads, len(d), ignore_garbage=False)[0] != Ld).sum()
    with_q = (C.decode_reads(reads, len(d), ignore_garbage=False, qual=q, qmin=20)[0] != Ld).sum()
    assert with_q < no_q

def test_end_to_end_rs_few_copies_custom_params():
    d = os.urandom(90000)
    for rs in [(42, 6), (20, 4)]:
        Ld, reads, q = _enc(d, 26, 0.001, rs)
        out, st = C.decode_reads(reads, len(d), rs, qual=q, qmin=10)
        assert C.letters_to_bytes(out, len(d)) == d and st["still_unreadable"] == 0
        out2, st2 = C.decode_reads(reads, len(d), rs, repair=False)
        assert st2["still_unreadable"] > 0
