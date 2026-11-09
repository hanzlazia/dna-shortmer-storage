# Combinatorial Shortmer DNA Storage
Based on: Preuss, Yakhini, Anavy - Data Storage Based on Combinatorial Synthesis of DNA Shortmers.

Run: `pip install -r requirements.txt && streamlit run app.py`

One page, step by step:
1. Upload a file (preview + metadata)
2. Optional compression (lossless LZMA, or lossy shrink for image / audio / video)
3. Synthesis: bits -> 12-bit letters -> each letter = 5 of the 16 trimers (Table 1) -> strands (barcode + 120 letters);
   optional Reed-Solomon (6 parity strands per 42 data strands, as in the paper); copies per strand; random seed
4. Substitution errors, shown molecule by molecule; download FASTQ / CSV
5. Decode (from the molecules above or an uploaded, edited FASTQ)
6. Compare original, compressed and recovered files

Files: `app.py` (interface), `shortmer_core.py` (encoding / decoding), `tests/` (`pytest tests`).
