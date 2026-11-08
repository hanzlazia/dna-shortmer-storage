import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from codec import *

def test_omega_distance_2():
    d = min(sum(a != b for a, b in zip(x, y)) for i, x in enumerate(OMEGA) for y in OMEGA[i + 1:])
    assert d == 2

def test_roundtrip():
    data = os.urandom(5000)
    assert letters_to_bytes(bytes_to_letters(data), len(data)) == data

def test_letter_mask_bijection():
    assert all(MASK_TO_LETTER[(1 << SUBSETS[i].astype(int)).sum()] == i for i in range(4096))

def test_simulate_deep_clean_is_exact():
    data = os.urandom(3000)
    L = bytes_to_letters(data)
    r = simulate(L, reads=300, p_err=0.0)
    assert letters_to_bytes(r["plain"], len(data)) == data

def test_check_beats_plain():
    data = os.urandom(20000)
    L = bytes_to_letters(data)
    r = simulate(L, reads=60, p_err=0.03)
    assert (r["checked"] != L).sum() < (r["plain"] != L).sum()
