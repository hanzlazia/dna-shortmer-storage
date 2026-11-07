# Combinatorial Shortmer DNA Storage

Interactive implementation of the **encoding** pipeline (plus error injection and decoding for verification) from

> I. Preuss, Z. Yakhini, L. Anavy, *Data Storage Based on Combinatorial Synthesis of DNA Shortmers*, bioRxiv 2021.08.01.454622.

Upload any file, see how it is turned into combinatorial DNA sequences, corrupt the stored data, and watch each error-correction layer repair it.

## Pipeline

| Step | Module | What it does |
|---|---|---|
| i | `shortmer/codec.py` | bytes → bits → combinatorial letters (binomial alphabet, ⌊log₂ C(N,K)⌋ bits/letter) |
| ii | `shortmer/codec.py`, `shortmer/rs.py` | lengthwise RS over GF(2^r) per sequence, crosswise RS per block, padding info, RS-protected barcodes (GF(16)) |
| iii | `shortmer/synthesis.py` | step-by-step shortmer synthesis protocol (mixing chamber, paper Fig. 2) |
| iv–v | `shortmer/channel.py` | molecular simulation (substitutions per nt, insertions/deletions per k-mer, read depth), grouping by barcode, top-K k-mers per position |
| vi–vii | `shortmer/codec.py` | errors-and-erasures RS decoding, lengthwise then crosswise, padding removal |

`shortmer/capacity.py` recomputes the paper's Table 3 (verified in the tests).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install pytest && pytest`

## Deploy (Streamlit Community Cloud)

Push this repo to GitHub, then at <https://share.streamlit.io> choose the repo, branch `main`, main file `app.py`.

## Notes and limits

* Binomial encoding is implemented end to end. The binary alphabet appears only in the capacity table.
* Letters are ranked lexicographically among K-subsets of Ω, so σ indices differ from the paper's arbitrary labels.
* The molecular channel follows the paper's description (Sec. 3.1) but is my own implementation; numbers will not match the paper's figures exactly.
* Input is limited to 200 KB to keep the demo interactive.
* The error-correction guarantees: lengthwise RS fixes ⌊parity/2⌋ unknown errors per sequence; crosswise RS fixes up to `block_parity` lost sequences per block.
