import numpy as np
import pytest

from shortmer import capacity
from shortmer.alphabet import PAPER_OMEGA_1, Alphabet, build_omega, min_hamming, rank, unrank
from shortmer.channel import inject_letter_errors, reconstruct, simulate_reads
from shortmer.codec import PRESETS, Params, decode, encode, make_alphabet
from shortmer.rs import GF, PRIMITIVE_POLY, RSCodec


@pytest.mark.parametrize("m", sorted(PRIMITIVE_POLY))
def test_fields_are_valid(m):
    GF.get(m)  # raises if the polynomial is not primitive


@pytest.mark.parametrize("m,n,nsym", [(4, 8, 2), (8, 40, 10), (9, 30, 6), (12, 46, 6), (13, 160, 14)])
def test_rs_errors_and_erasures(m, n, nsym):
    rng = np.random.default_rng(m * 100 + n)
    rs = RSCodec(m, nsym)
    q = 1 << m
    for _ in range(60):
        msg = [int(x) for x in rng.integers(0, q, n - nsym)]
        cw = rs.encode(msg)
        assert not any(rs.syndromes(cw))
        e = int(rng.integers(0, nsym + 1))            # erasures
        t = int(rng.integers(0, (nsym - e) // 2 + 1))  # errors within budget
        pos = rng.choice(n, size=e + t, replace=False)
        bad = list(cw)
        for j in pos:
            bad[j] = int((bad[j] + rng.integers(1, q)) % q) if bad[j] + 1 else 0
            if bad[j] == cw[j]:
                bad[j] = (cw[j] + 1) % q
        fixed, _ = rs.decode(bad, erasures=[int(j) for j in pos[:e]])
        assert fixed == cw


def test_rs_failure_is_reported():
    rs = RSCodec(8, 4)
    cw = rs.encode(list(range(30)))
    bad = list(cw)
    for j in range(6):
        bad[j] ^= 0x55
    fixed, _ = rs.decode(bad)
    assert fixed is None or fixed != cw


def test_paper_omega_distance():
    assert len(PAPER_OMEGA_1) == 16 and min_hamming(PAPER_OMEGA_1) == 2
    assert min_hamming(build_omega(4, 20, 2)) >= 2
    with pytest.raises(ValueError):
        build_omega(3, 20, 2)  # Singleton bound: at most 4^(3-2+1) = 16 trimers


def test_rank_unrank_roundtrip():
    for idx in (0, 1, 100, 4367):
        assert rank(unrank(idx, 16, 5), 16, 5) == idx
    assert unrank(0, 16, 5) == (0, 1, 2, 3, 4)


def test_alphabet_sizes_match_paper():
    a = Alphabet(PAPER_OMEGA_1, 5)
    assert a.size == 4368 and a.usable == 4096 and a.bits == 12
    assert Alphabet(PAPER_OMEGA_1, 3).bits == 9 and Alphabet(PAPER_OMEGA_1, 7).bits == 13


def test_capacity_table_matches_paper_table3():
    t = capacity.table3().set_index(["Type", "N", "K"], drop=False)
    r = capacity.capacity_row("Binomial", 16, 5)
    assert r["Bits / sequence"] == 1440 and r["Bits / cycle"] == 8.4 and r["Fold increase"] == 6.0
    r = capacity.capacity_row("Standard", None, None)
    assert r["Bits / cycle"] == 1.4 and r["Bits / cycle (payload only)"] == 1.57
    r = capacity.capacity_row("Binary", 20, None)
    assert r["Bits / cycle"] == 13.3 and r["Fold increase"] == 9.5
    assert len(t) == 7


@pytest.mark.parametrize("preset", list(PRESETS))
@pytest.mark.parametrize("size", [1, 37, 1000])
def test_clean_roundtrip(preset, size):
    p = PRESETS[preset]
    data = bytes(np.random.default_rng(size).integers(0, 256, size, dtype=np.uint8))
    enc = encode(data, p)
    rep = decode([list(r) for r in enc.matrix], p, enc.alphabet)
    assert rep.ok and rep.recovered == data


def test_parity_rows_are_lengthwise_codewords():
    p = Params()
    enc = encode(b"hello world" * 20, p)
    rs = RSCodec(enc.alphabet.bits, p.rs_len)
    assert all(not any(rs.syndromes(r)) for r in enc.matrix)


def test_injection_recovery_layers():
    p = Params()
    data = bytes(range(256)) * 3
    enc = encode(data, p)
    rng = np.random.default_rng(1)
    # within lengthwise capacity: 3 errors per row on a few rows
    inj = inject_letter_errors(enc, rng, burst_row=2, burst_len=3)
    rep = decode(inj.received, p, enc.alphabet)
    assert rep.ok and rep.recovered == data and rep.stats["crosswise_symbols_fixed"] == 0
    # beyond lengthwise capacity, repaired by crosswise RS
    inj = inject_letter_errors(enc, rng, burst_row=2, burst_len=12, lose_rows=1)
    rep = decode(inj.received, p, enc.alphabet)
    assert rep.ok and rep.recovered == data and rep.stats["lengthwise_failed"] >= 1
    # too many lost sequences in one block -> reported as failure, never silently wrong
    inj = inject_letter_errors(enc, rng, lose_rows=enc.total_rows - 2)
    rep = decode(inj.received, p, enc.alphabet)
    assert not rep.ok or rep.recovered == data


def test_molecular_channel_clean_and_noisy():
    p = Params()
    data = b"DNA data storage with combinatorial shortmers. " * 12
    enc = encode(data, p)
    rng = np.random.default_rng(7)
    reads, st = simulate_reads(enc, rng, depth=60, p_sub=0, p_ins=0, p_del=0)
    mat, st = reconstruct(reads, p, enc.alphabet, st, enc.total_rows)
    assert mat == [list(r) for r in enc.matrix]
    reads, st = simulate_reads(enc, rng, depth=80, p_sub=0.005, p_ins=0.002, p_del=0.002, lose_rows=1)
    mat, st = reconstruct(reads, p, enc.alphabet, st, enc.total_rows)
    rep = decode(mat, p, enc.alphabet)
    assert rep.ok and rep.recovered == data
