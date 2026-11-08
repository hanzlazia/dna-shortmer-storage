"""Encode a file into combinatorial shortmer DNA (Preuss, Yakhini & Anavy 2021), download it, edit it, decode it."""
import hashlib
import io
import lzma
import math
import mimetypes
import numpy as np
import pandas as pd
import streamlit as st
import dna_codec as C

st.set_page_config(page_title="Shortmer DNA Storage", layout="wide")
MAX_READS = 200_000          # molecules kept in memory / in one FASTQ download

# ------------------------------------------------------------------ small helpers / drawings
COL = [f"hsl({int(i * 22.5)},70%,80%)" for i in range(C.N)]


def chip(t: str, good: bool = True) -> str:
    bg = COL[C.OMEGA.index(t)] if t in C.OMEGA else "#f4b6b6"
    return f"<span style='background:{bg};border-radius:4px;padding:1px 5px;margin:1px;font-family:monospace;font-size:13px'>{t}</span>"


def helix_svg(w=1200, h=64) -> str:
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}'>"]
    pair = ["#e07a5f", "#3d5a80", "#81b29a", "#f2cc8f"]
    for i, x in enumerate(range(10, w - 10, 12)):
        ph = x / 38.0
        y1, y2 = h / 2 + 24 * math.sin(ph), h / 2 - 24 * math.sin(ph)
        out.append(f"<line x1='{x}' y1='{y1:.1f}' x2='{x}' y2='{y2:.1f}' stroke='{pair[i % 4]}' stroke-width='3' opacity='.7'/>")
    for sgn, c in ((1, "#33414f"), (-1, "#8a99a8")):
        pts = " ".join(f"{x},{h / 2 + sgn * 24 * math.sin(x / 38.0):.1f}" for x in range(10, w - 10, 4))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='3'/>")
    return "".join(out) + "</svg>"


def pipeline_svg() -> str:
    steps = ["File", "Bits\n0101…", "12-bit\nletters", "Letter = 5 of\n16 trimers", "Strand = barcode\n+ 120 letters",
             "Molecules\n(copies)", "FASTQ / CSV", "Decode:\nvote → file"]
    w, bw = 1200, 150
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} 70' font-family='serif' font-size='13'>"]
    for i, s in enumerate(steps):
        x = 4 + i * (bw)
        fill = "#33414f" if i not in (6,) else "#3d7a5a"
        out.append(f"<polygon points='{x},4 {x + bw - 14},4 {x + bw},35 {x + bw - 14},66 {x},66 {x + 14},35' fill='{fill}'/>")
        l = s.split("\n")
        for j, t in enumerate(l):
            y = 35 + (j - (len(l) - 1) / 2) * 16 + 4
            out.append(f"<text x='{x + bw / 2 + 2}' y='{y}' fill='white' text-anchor='middle'>{t}</text>")
    return "".join(out) + "</svg>"


def preview(data: bytes, name: str):
    mt = mimetypes.guess_type(name)[0] or ""
    kind = mt.split("/")[0]
    try:
        if kind == "image":
            st.image(data, width='stretch')
        elif kind == "audio":
            st.audio(data, format=mt)
        elif kind == "video":
            st.video(data, format=mt)
        elif kind == "text" or name.lower().endswith((".json", ".csv", ".md", ".py", ".txt")):
            st.code(data[:3000].decode("utf-8", errors="replace"))
        else:
            st.caption("No preview for this file type - use the download button.")
    except Exception:
        st.warning("This file cannot be previewed (it may be damaged).")


def meta_table(name: str, data: bytes):
    st.dataframe(pd.DataFrame({"Field": ["Name", "Type", "Size", "SHA-256 (first 16)"],
                               "Value": [name, mimetypes.guess_type(name)[0] or "unknown", f"{len(data):,} bytes",
                                         hashlib.sha256(data).hexdigest()[:16]]}), hide_index=True, width='stretch')


@st.cache_data(max_entries=2, show_spinner="Synthesizing molecules...")
def get_clean(sha: str, _letters, copies: int, seed: int):
    return C.make_reads(_letters, copies, 0.0, seed)[0]


@st.cache_data(max_entries=2, show_spinner="Introducing errors...")
def get_noisy(sha: str, copies: int, seed: int, p: float, _clean):
    return C.add_substitutions(_clean, p, seed)


def shrink_image(data: bytes, max_side: int, quality: int) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def viewer(reads, letters, copies, sidx, npos, nmol):
    mol = reads[sidx * copies: sidx * copies + nmol]
    lt = letters[sidx * C.LPS: sidx * C.LPS + npos]
    H = ["<table style='border-collapse:collapse;font-size:13px'>",
         "<tr><td></td><td><b>barcode</b></td>" + "".join(f"<td style='padding:0 8px'><b>pos {i}</b></td>" for i in range(npos)) + "</tr>",
         "<tr><td style='padding-right:8px'><b>Letter (mixture)</b></td><td></td>" +
         "".join("<td style='padding:2px 8px;border-left:1px solid #ccc'>" + "<br>".join(chip(t) for t in C.letter_trimers(int(v))) + "</td>" for v in lt) + "</tr>"]
    for k, r in enumerate(mol):
        x = r.tobytes().decode()
        H.append(f"<tr><td style='padding-right:8px'>molecule {k}</td><td><code>{x[:12]}</code></td>" +
                 "".join(f"<td style='padding:2px 8px;border-left:1px solid #ccc'>{chip(x[12 + 3 * i:15 + 3 * i])}</td>" for i in range(npos)) + "</tr>")
    st.markdown("".join(H) + "</table>", unsafe_allow_html=True)


def step(n, title):
    st.markdown(f"## Step {n} · {title}")



# ------------------------------------------------------------------ page
st.title("Combinatorial Shortmer DNA Storage")
st.image(helix_svg(), width='stretch')
st.image(pipeline_svg(), width='stretch')

# ---------------------------------------------------------------- 1 upload
step(1, "Upload your file")
up = st.file_uploader("Any file: image, audio, video, text, ...", label_visibility="collapsed")
if up is None:
    orig, oname = b"DNA storage with combinatorial shortmers.", "sample.txt"
    st.caption("No file uploaded: a short sample text is used.")
else:
    orig, oname = up.getvalue(), up.name
if not orig:
    st.warning("The file is empty."); st.stop()
c1, c2 = st.columns(2)
with c1: preview(orig, oname)
with c2: meta_table(oname, orig)

# ---------------------------------------------------------------- 2 compress
step(2, "Compress the file (optional)")
is_img = (mimetypes.guess_type(oname)[0] or "").startswith("image")
modes = ["No compression", "Lossless compression (LZMA) - any file, exact recovery"]
if is_img:
    modes.append("Shrink the image (lossy JPEG) - smaller picture, much fewer DNA strands")
mode = st.radio("Compression", modes, label_visibility="collapsed")
payload, pname, comp = orig, oname, "none"
if mode.startswith("Lossless"):
    payload, comp = lzma.compress(orig, preset=9 | lzma.PRESET_EXTREME), "lzma"
elif mode.startswith("Shrink"):
    q1, q2 = st.columns(2)
    side = q1.slider("Longest side (pixels)", 64, 3000, 640)
    qual = q2.slider("JPEG quality", 5, 95, 40)
    try:
        payload = shrink_image(orig, side, qual)
        pname = oname.rsplit(".", 1)[0] + "_small.jpg"
    except Exception:
        st.error("This image could not be opened for shrinking."); payload = orig
psha = sha16(payload)
k1, k2, k3 = st.columns(3)
k1.metric("Original", f"{len(orig):,} bytes")
k2.metric("To be stored in DNA", f"{len(payload):,} bytes", f"{(len(payload) / len(orig) - 1):+.0%}" if len(payload) != len(orig) else None, delta_color="off")
k3.metric("Bits", f"{len(payload) * 8:,}")
if mode.startswith("Shrink"):
    preview(payload, pname)
b1, b2 = st.columns(2)
b1.download_button("Download binary (.bin)", payload, file_name=pname + ".bin", on_click="ignore", help="The bytes that will be written into DNA.")
if len(payload) <= 3_000_000:
    b2.download_button("Download bits (0/1 text)", lambda: "".join(map(str, np.unpackbits(np.frombuffer(payload, np.uint8)))).encode(),
                       file_name=pname + "_bits.txt", on_click="ignore")

letters = C.bytes_to_letters(payload)
n_letters, n_strands = len(letters), len(letters) // C.LPS
m1, m2, m3 = st.columns(3)
m1.metric("Letters (12 bits each)", f"{n_letters:,}"); m2.metric("Strands (120 letters)", f"{n_strands:,}")
m3.metric("Bits per nucleotide", f"{C.BITS / 3:.0f}", help="Standard DNA: 2")

with st.expander("The encoding, letter by letter (first letters of the file)", expanded=True):
    bits = np.unpackbits(np.frombuffer(payload[:10], dtype=np.uint8))
    rows = [{"Position": i, "12 bits": "".join(map(str, np.pad(bits, (0, 96))[12 * i:12 * i + 12])), "Letter number": int(letters[i]),
             "Letter = 5 trimers": " ".join(C.letter_trimers(int(letters[i]))),
             "As Omega indices": " ".join(f"X{j + 1}" for j in C.SUBSETS[letters[i]])} for i in range(6)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.markdown("Omega (paper Table 1): " + " ".join(f"<small>X{i + 1}</small>{chip(t)}" for i, t in enumerate(C.OMEGA)), unsafe_allow_html=True)

# ---------------------------------------------------------------- 3 synthesis
step(3, "Synthesis: make the molecules")
s1, s2 = st.columns(2)
copies = s1.slider("Copies (molecules) per strand", 1, 300, 80,
                   help="Each letter is a mixture of 5 trimers, but each molecule carries only ONE of them at every position, so many copies are needed to see all 5.")
seed = s2.number_input("Random seed", 0, 10_000, 1,
                       help="The molecules pick their trimers at random. The seed fixes that randomness: the same seed gives exactly the same molecules and errors every time; a different seed gives a different random outcome.")
n_reads = n_strands * copies
miss = C.miss_probability(copies)
st.caption(f"**{n_reads:,}** molecules. A letter misses at least one of its 5 trimers with probability **{miss:.2%}** → about "
           f"**{miss * n_letters:,.0f}** of {n_letters:,} letters unreadable even with no errors (≈ 11.4 copies needed on average).")
if n_reads > MAX_READS:
    st.error(f"{n_reads:,} molecules is more than this app can handle ({MAX_READS:,}). With {copies} copies the file to store must be at most "
             f"**{MAX_READS // copies * C.LPS * 12 // 8:,} bytes**. Use Step 2 (compression / shrink image) or fewer copies.")
    st.stop()
clean = get_clean(psha, letters, copies, int(seed))
v1, v2, v3 = st.columns(3)
sidx = v1.number_input("Strand number", 0, n_strands - 1, 0)
npos = v2.slider("Positions to show", 3, 16, 8)
nmol = v3.slider("Molecules to show", 1, 15, min(8, copies))
st.markdown("**Top row:** the letter at each position = its 5 trimers (the mixture that is synthesized). "
            "**Rows below:** real molecules; each one holds a single trimer of the 5 at every position.")
viewer(clean, letters, copies, sidx, npos, nmol)

# ---------------------------------------------------------------- 4 errors
step(4, "Introduce substitution errors")
p_sub = st.slider("Substitution error per nucleotide (%)", 0.0, 5.0, 0.5, 0.1,
                  help="Each base of each molecule is replaced by a different base with this probability.") / 100
reads, n_changed = get_noisy(psha, copies, int(seed), p_sub, clean)
st.caption(f"**{n_changed:,}** of {reads.size:,} bases changed. Red chips = the trimer is no longer in Omega.")
if p_sub > 0:
    viewer(reads, letters, copies, sidx, npos, nmol)
meta = dict(name=pname if comp == "none" else oname, bytes=len(payload), sha=psha, comp=comp)
if comp == "lzma":
    meta["orig"] = sha16(orig)
d1, d2, d3 = st.columns(3)
mb = len(reads) * (2 * C.READ_LEN + 120) / 1e6
d1.download_button(f"Download FASTQ ({mb:,.0f} MB)", lambda: C.to_fastq(reads, copies, meta), file_name="encoded.fastq",
                   mime="text/plain", on_click="ignore", help="The molecules (with these errors). Edit it by hand if you like, then upload it in Step 5.")
d2.download_button("Download CSV: molecules", lambda: C.to_molecule_csv(reads, copies), file_name="molecules.csv",
                   mime="text/csv", on_click="ignore", help="One row per molecule, trimers separated by spaces.")
if n_letters <= 300_000:
    d3.download_button("Download CSV: encoding table", lambda: C.to_letter_csv(letters), file_name="encoding_table.csv",
                       mime="text/csv", on_click="ignore")

# ---------------------------------------------------------------- 5 decode
step(5, "Decode")
src = st.radio("Molecules to decode", ["Use the molecules from Step 4 (no upload needed)", "Upload a FASTQ file (for example the downloaded one, edited by hand)"])
correct = st.checkbox("Error-correcting decoder", True,
                      help="ON: barcode repaired, reads with a trimer outside Omega are ignored, the 5 most frequent trimers form the letter. "
                           "OFF: any unexpected trimer makes the letter unreadable.")
reads_in, dmeta, dropped = None, {}, 0
if src.startswith("Use"):
    reads_in, dmeta = reads, meta
else:
    fq = st.file_uploader("FASTQ (.fastq or .fastq.gz)", key="fq")
    if fq is not None:
        reads_in, dmeta, dropped = C.parse_fastq(fq.getvalue())
        if len(reads_in) == 0:
            st.error("No reads of the expected length (372 nt) found."); reads_in = None
key = (psha, copies, int(seed), p_sub, correct, src[:3], getattr(fq, "file_id", None) if not src.startswith("Use") else None)
if reads_in is not None and st.button("Decode", type="primary"):
    with st.spinner("Decoding..."):
        let, stt = C.decode_reads(reads_in, dmeta.get("bytes"), correct)
        nb = dmeta.get("bytes") or len(let) * C.BITS // 8
        st.session_state["dec"] = dict(out=C.letters_to_bytes(let, nb), stt=stt, meta=dmeta, dropped=dropped, correct=correct, key=key)
dec = st.session_state.get("dec")
if dec and dec["key"] == key:
    stt, dm, out = dec["stt"], dec["meta"], dec["out"]
    same = bool(dm.get("sha")) and sha16(out) == dm["sha"]
    (st.success if same else st.error)("Recovered bytes are IDENTICAL to what was written (SHA-256 matches)." if same else
                                      "Recovered bytes differ from what was written (or the FASTQ header has no checksum).")
    a, b, c, d = st.columns(4)
    a.metric("Molecules read", f"{stt['reads_in']:,}", f"-{dec['dropped']:,} wrong length" if dec["dropped"] else None, delta_color="off")
    b.metric("Reads dropped (bad barcode)", f"{stt['bad_barcode']:,}")
    c.metric("Trimer reads ignored (not in Omega)", f"{stt['ignored_trimers']:,}" if dec["correct"] else "n/a")
    d.metric("Unreadable letters", f"{stt['unreadable_letters']:,} / {stt['letters']:,}")
    st.caption(f"Strands with no molecule: {stt['strands_missing']:,} · unreadable letters are written as zeros.")

    # ------------------------------------------------------------ 6 decompress + preview
    step(6, "Decompress and compare")
    final, fname = out, dm.get("name", "decoded.bin")
    if dm.get("comp") == "lzma":
        try:
            final = lzma.decompress(out)
            st.success(f"Decompressed: {len(out):,} → {len(final):,} bytes" +
                       (" (matches the original checksum)" if dm.get("orig") == sha16(final) else " (checksum does NOT match the original)"))
        except Exception:
            st.error("The compressed data is damaged and cannot be decompressed (too many unreadable letters). Showing the raw bytes instead.")
            fname = fname + ".lzma"
    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Original"); preview(orig, oname); meta_table(oname, orig)
    with p2:
        st.subheader("Recovered"); preview(final, fname); meta_table(fname, final)
    st.download_button("Download recovered file", final, file_name="recovered_" + fname, on_click="ignore")
    if dm.get("comp") == "none" and mode.startswith("Shrink") and src.startswith("Use"):
        st.caption("The recovered file is the shrunk picture from Step 2 (lossy shrinking cannot be undone), so it looks smaller than the original.")
