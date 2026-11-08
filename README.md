# Shortmer DNA storage (Preuss, Yakhini, Anavy 2021)
Run: `pip install -r requirements.txt && streamlit run app.py`
Tab 1 encodes a file (bits -> 12-bit letters -> 5 of 16 trimers -> strands -> molecules) and offers FASTQ/CSV downloads.
Tab 2 decodes the in-app molecules or an uploaded (edited) FASTQ. Tests: `pytest tests`.
