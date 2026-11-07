"""Step-by-step synthesis protocol for one sequence (paper Sec. 2.1, Figure 2)."""
from __future__ import annotations

from dataclasses import dataclass

from .alphabet import Alphabet


@dataclass
class Step:
    n: int
    phase: str          # Linker | Barcode | Letter i
    action: str
    chamber: list[str]  # contents of the mixing chamber after the step
    strand: str         # consensus of the elongating strand (letters shown as {..})


def protocol(barcode: str, letters: list[int], alpha: Alphabet) -> list[Step]:
    steps: list[Step] = [Step(1, "Linker", "Start from a linker attached to the solid surface.", [], "")]
    strand = ""
    n = 1
    for nt in barcode:
        n += 1
        strand += nt
        steps.append(Step(n, "Barcode", f"Inject standard phosphoramidite {nt} and couple (elongation).", [], strand))
    for pos, v in enumerate(letters):
        ks = alpha.kmers(v)
        chamber: list[str] = []
        for km in ks:
            n += 1
            chamber.append(km)
            steps.append(Step(n, f"Letter {pos}", f"Inject {km} into the mixing chamber.", list(chamber), strand))
        n += 1
        steps.append(Step(n, f"Letter {pos}", f"Mix the {len(ks)} k-mers.", list(chamber), strand))
        n += 1
        strand += "{" + ",".join(ks) + "}"
        steps.append(Step(n, f"Letter {pos}", "Inject the mixture onto the surface; each growing strand "
                          "couples one k-mer chosen at random (elongation).", [], strand))
    steps.append(Step(n + 1, "Cleavage", "Cleave the molecules from the surface and collect them.", [], strand))
    return steps


def cycle_counts(barcode_len: int, n_letters: int, K: int) -> dict[str, int]:
    """Dispense operations for one sequence vs. standard DNA with equal capacity."""
    return {
        "barcode couplings": barcode_len,
        "k-mer injections (to mixing chamber)": n_letters * K,
        "mix steps": n_letters,
        "elongation steps": barcode_len + n_letters,
    }
