"""Encoding walk-through of: Preuss, Yakhini, Anavy - Data Storage Based on Combinatorial Synthesis of DNA Shortmers."""
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import codec as C

st.set_page_config(page_title="Shortmer DNA Encoding", layout="wide")
st.title("Combinatorial Shortmer DNA Encoding")
st.caption("Binomial encoding of Preuss, Yakhini & Anavy (2021): Omega = 16 trimers (Table 1), K = 5 trimers per "
           "letter, 4096 letters = 12 bits per letter. All parameters are fixed to the paper.")

up = st.file_uploader("Upload any file", help="Any file type. The file is converted to bits, then to DNA letters.")
if up is None:
    st.info("Upload a file to start (or the sample text below is used).")
    data, name = b"DNA storage with combinatorial shortmers.", "sample.txt"
else:
    data, name = up.getvalue(), up.name
if not data:
    st.warning("The file is empty.")
    st.stop()

with st.spinner("Encoding..."):
    letters = C.bytes_to_letters(data)
n_letters = len(letters)
n_strands = n_letters // C.LETTERS_PER_STRAND
n_bits = len(data) * 8

a, b, c, d = st.columns(4)
a.metric("File size", f"{len(data):,} bytes")
b.metric("Bits", f"{n_bits:,}")
c.metric("Letters (12 bits each)", f"{n_letters:,}")
d.metric("DNA strands", f"{n_strands:,}")

# ---------------------------------------------------------------- the encoding, step by step
st.header("1. How the file becomes DNA")
st.markdown(
    f"**Step 1 - file to bits.** `{name}` is read as bytes and written as 0/1 bits.  \n"
    f"**Step 2 - bits to letters.** Every **12 bits** are one number from 0 to 4095 (one *combinatorial letter*).  \n"
    f"**Step 3 - letter to 5 trimers.** Each letter is a set of **5 trimers** out of the 16 in Omega "
    f"(there are C(16,5) = 4368 such sets, 4096 are used). Different letter = different set.  \n"
    f"**Step 4 - strand.** {C.LETTERS_PER_STRAND} letters are put after a {C.BARCODE_LEN}-nt standard-DNA barcode "
    f"(the strand number) so that the strands can be put back in order.  \n"
    f"**Step 5 - synthesis.** At each letter position all 5 trimers are mixed, so the many copies of one strand "
    f"are *different molecules*: each molecule has just one of the 5 trimers at that position.")

st.subheader("Omega (paper Table 1)")
st.dataframe(pd.DataFrame({"": [f"X{i+1}" for i in range(C.N)], "trimer": C.OMEGA}).set_index("").T, use_container_width=True)

st.subheader("The first letters of your file")
bits = np.unpackbits(np.frombuffer(data[:max(1, min(len(data), 60))], dtype=np.uint8))
rows = []
for i in range(min(8, n_letters)):
    v = int(letters[i])
    rows.append({"Letter #": i + 1,
                 "12 bits": "".join(map(str, np.pad(bits, (0, 12 * 8))[12 * i:12 * i + 12])),
                 "Number": v,
                 "Set of 5 trimers": "{" + ", ".join(C.letter_trimers(v)) + "}",
                 "In Omega terms": "{" + ", ".join(f"X{j+1}" for j in C.SUBSETS[v]) + "}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("The first strands")
show = st.slider("Letters to show per strand", 3, 20, 8)
srows = []
for s in range(min(4, n_strands)):
    row = letters[s * C.LETTERS_PER_STRAND:(s + 1) * C.LETTERS_PER_STRAND]
    srows.append({"Strand": s, "Barcode (standard DNA)": C.barcode(s),
                  "Payload letters": "  ".join("{" + ",".join(C.letter_trimers(int(v))) + "}" for v in row[:show]) + "  ..."})
st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)

st.subheader("What is actually synthesized: different molecules of the same strand")
st.caption("Strand 0, first letters. Same data, but each molecule randomly carries one of the 5 trimers of every letter.")
mols = C.example_molecules(letters[:C.LETTERS_PER_STRAND][:show], n=6)
bc = C.barcode(0)
st.code("\n".join(f"{bc} {' '.join(m[i:i+3] for i in range(0, len(m), 3))}" for m in mols))

d1, d2, d3 = st.columns(3)
d1.metric("Bits per letter", C.BITS)
d2.metric("Bits per nucleotide (payload)", f"{C.BITS / 3:.0f}", help="12 bits are carried by one trimer position (3 nt); standard DNA gives 2 bits per nt")
d3.metric("Reads to see all 5 trimers", "≈ 11.4", help="Coupon collector: K*H(K) = 5 * (1 + 1/2 + 1/3 + 1/4 + 1/5)")

# ---------------------------------------------------------------- reading back
st.header("2. Reading it back: without and with errors")
st.markdown("To read a letter we sequence many molecules and collect the trimers we see at that position; "
            "the 5 trimers seen give the letter back. Errors are single wrong bases inside a read.")
e1, e2 = st.columns(2)
reads = e1.slider("Reads per letter position", 10, 200, 40)
p_err = e2.slider("Error probability per read (wrong base)", 0.0, 0.10, 0.02, 0.005)

res = C.simulate(letters, reads, p_err)
orig_sha = hashlib.sha256(data).hexdigest()
table, outs = [], {}
for key, label in [("clean", "No errors"), ("plain", "With errors, plain decoding"),
                   ("checked", "With errors + paper's d = 2 check (ignore trimers not in Omega)")]:
    out = C.letters_to_bytes(res[key], len(data))
    outs[key] = out
    wrong = int((res[key] != letters).sum())
    table.append({"Case": label, "Letters wrong": f"{wrong:,} / {n_letters:,}",
                  "Bytes wrong": f"{sum(x != y for x, y in zip(out, data)) if len(data) < 50_000 else int((np.frombuffer(out, np.uint8) != np.frombuffer(data, np.uint8)).sum()):,}",
                  "File identical": "YES" if hashlib.sha256(out).hexdigest() == orig_sha else "no"})
st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
st.caption("Why: every error makes a trimer that is not in Omega (minimum Hamming distance 2 means one wrong base "
           "can never turn a trimer into another member of Omega), so it can be detected and ignored. "
           "Unreadable letters are replaced by zeros. Only substitutions are modelled (not insertions/deletions).")

if name.lower().endswith((".txt", ".md", ".csv", ".json", ".py")):
    st.subheader("Text preview (first 300 characters)")
    for key, label in [("clean", "No errors"), ("plain", "With errors, plain"), ("checked", "With errors + d = 2 check")]:
        st.markdown(f"**{label}**")
        st.code(outs[key][:300].decode("utf-8", errors="replace"))
st.download_button("Download decoded file (errors + d = 2 check)", outs["checked"], file_name="decoded_" + name)
