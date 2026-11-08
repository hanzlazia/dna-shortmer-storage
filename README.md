# Shortmer DNA encoding (Preuss, Yakhini, Anavy 2021)
Run: `pip install -r requirements.txt && streamlit run app.py`
File -> bits -> 12-bit letters -> each letter = set of 5 of the 16 trimers (Table 1) -> strands (12-nt barcode + 120 letters).
Then: read back with/without substitution errors (d = 2 check). Tests: `pytest tests`.
