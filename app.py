"""Encode a file into combinatorial shortmer DNA (Preuss, Yakhini & Anavy 2021), download it, edit it, decode it."""
import hashlib
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
def get_reads(sha: str, _letters, copies: int, p: float, seed: int):
    return C.make_reads(_letters, copies, p, seed)


# ------------------------------------------------------------------ header
st.title("Combinatorial Shortmer DNA Storage")
st.image(helix_svg(), width='stretch')
st.image(pipeline_svg(), width='stretch')
tab_enc, tab_dec = st.tabs(["1 · Encode", "2 · Decode"])

# ================================================================== ENCODE
with tab_enc:
    up = st.file_uploader("Upload any file (image, audio, video, text, ...)")
    if up is None:
        data, name = b"DNA storage with combinatorial shortmers.", "sample.txt"
        st.caption("No file uploaded: a short sample text is used.")
    else:
        data, name = up.getvalue(), up.name
    if not data:
        st.warning("The file is empty."); st.stop()
    sha = hashlib.sha256(data).hexdigest()
    letters = C.bytes_to_letters(data)
    n_letters, n_strands = len(letters), len(letters) // C.LPS

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Your file"); preview(data, name)
    with c2:
        st.subheader("File metadata"); meta_table(name, data)
        m1, m2 = st.columns(2)
        m1.metric("Bits", f"{len(data) * 8:,}"); m2.metric("Letters (12 bits)", f"{n_letters:,}")
        m1.metric("Strands (120 letters)", f"{n_strands:,}"); m2.metric("Bits per nucleotide", f"{C.BITS / 3:.0f}", help="Standard DNA: 2")

    st.subheader("Omega: the 16 trimers (paper Table 1)")
    st.markdown(" ".join(f"<small>X{i + 1}</small>{chip(t)}" for i, t in enumerate(C.OMEGA)), unsafe_allow_html=True)

    st.subheader("Synthesis settings")
    s1, s2, s3 = st.columns(3)
    copies = s1.slider("Copies (molecules) per strand", 1, 300, 60,
                       help="Each letter has 5 trimers. Each molecule carries only ONE of them at every position, so many copies are needed to see all 5.")
    p_sub = s2.slider("Substitution error per nucleotide (%)", 0.0, 5.0, 0.5, 0.1) / 100
    seed = s3.number_input("Random seed", 0, 10_000, 1)
    n_reads = n_strands * copies
    miss = C.miss_probability(copies)
    st.caption(f"**{n_reads:,}** molecules in total.  A letter misses at least one of its 5 trimers with probability "
               f"**{miss:.2%}**  →  expect about **{miss * n_letters:,.0f}** of {n_letters:,} letters to be unreadable even without errors "
               f"(≈ {C.K * sum(1 / i for i in range(1, C.K + 1)):.1f} copies are needed on average, more for safety).")
    too_big = n_reads > MAX_READS
    if too_big:
        st.error(f"{n_reads:,} molecules is too many for this app (limit {MAX_READS:,}). Lower the copies or use a smaller file "
                 f"(at most {MAX_READS // max(copies, 1) * C.LPS * 12 // 8:,} bytes with {copies} copies).")
        reads = strand_ids = None
    else:
        reads, strand_ids = get_reads(sha, letters, copies, p_sub, int(seed))
        st.session_state["enc"] = dict(sha=sha, name=name, data=data, copies=copies, reads=reads)

    st.subheader("The encoding, letter by letter")
    bits = np.unpackbits(np.frombuffer(data[:10], dtype=np.uint8))
    rows = [{"Position": i, "12 bits": "".join(map(str, np.pad(bits, (0, 96))[12 * i:12 * i + 12])), "Letter number": int(letters[i]),
             "Letter = 5 trimers": " ".join(C.letter_trimers(int(letters[i]))),
             "As Omega indices": " ".join(f"X{j + 1}" for j in C.SUBSETS[letters[i]])} for i in range(6)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    st.subheader("A strand: the letter (5 trimers) and the molecules made from it")
    v1, v2, v3 = st.columns(3)
    sidx = v1.number_input("Strand number", 0, n_strands - 1, 0)
    npos = v2.slider("Positions to show", 3, 16, 8)
    nmol = v3.slider("Molecules to show", 1, 15, min(8, copies))
    if reads is not None:
        mol = reads[sidx * copies: sidx * copies + nmol]
    else:
        mol, _ = C.make_reads(letters[sidx * C.LPS:(sidx + 1) * C.LPS], copies, p_sub, int(seed))
        mol = mol[:nmol]; st.caption("(sample only - not part of any download)")
    lt = letters[sidx * C.LPS: sidx * C.LPS + npos]
    H = ["<table style='border-collapse:collapse;font-size:13px'>",
         "<tr><td></td><td><b>barcode</b></td>" + "".join(f"<td style='padding:0 8px'><b>pos {i}</b></td>" for i in range(npos)) + "</tr>",
         "<tr><td style='padding-right:8px'><b>Letter (mixture)</b></td><td></td>" +
         "".join("<td style='padding:2px 8px;border-left:1px solid #ccc'>" + "<br>".join(chip(t) for t in C.letter_trimers(int(v))) + "</td>" for v in lt) + "</tr>"]
    for k, r in enumerate(mol):
        s = r.tobytes().decode()
        H.append(f"<tr><td style='padding-right:8px'>molecule {k}</td><td><code>{s[:12]}</code></td>" +
                 "".join(f"<td style='padding:2px 8px;border-left:1px solid #ccc'>{chip(s[12 + 3 * i:15 + 3 * i])}</td>" for i in range(npos)) + "</tr>")
    st.markdown("".join(H) + "</table>", unsafe_allow_html=True)
    st.caption("Each molecule picks one trimer out of the letter's 5 at every position. Red = a trimer that is not in Omega (substitution error).")

    st.subheader("Download the encoded file")
    meta_fq = (name, len(data), sha[:16])
    d1, d2, d3 = st.columns(3)
    if reads is not None:
        mb = n_reads * (2 * C.READ_LEN + 90) / 1e6
        d1.download_button(f"FASTQ ({mb:,.0f} MB)", lambda: C.to_fastq(reads, copies, *meta_fq), file_name="encoded.fastq",
                           mime="text/plain", on_click="ignore", help="Molecules with metadata in the header lines. Edit it, then decode it in tab 2.")
        d2.download_button("CSV: molecules", lambda: C.to_molecule_csv(reads, copies), file_name="molecules.csv",
                           mime="text/csv", on_click="ignore", help="One row per molecule, trimers separated by spaces.")
    if n_letters <= 300_000:
        d3.download_button("CSV: encoding table", lambda: C.to_letter_csv(letters), file_name="encoding_table.csv",
                           mime="text/csv", on_click="ignore", help="bits → letter → 5 trimers for every position")
    else:
        d3.caption("Encoding table CSV is offered for files up to about 450 KB.")

# ================================================================== DECODE
with tab_dec:
    src = st.radio("Molecules to decode", ["The molecules just made in tab 1", "Upload a FASTQ file (e.g. the downloaded one, after editing)"])
    correct = st.checkbox("Error-correcting decoder", True,
                          help="ON: barcode repaired, reads with a trimer outside Omega are ignored, the 5 most frequent trimers form the letter. "
                               "OFF: any unexpected trimer makes the letter unreadable.")
    reads_in, meta, dropped = None, {}, 0
    enc = st.session_state.get("enc")
    if src.startswith("The"):
        if enc is None:
            st.info("Encode a file in tab 1 first (with a molecule count within the limit).")
        else:
            reads_in = enc["reads"]
            meta = dict(name=enc["name"], bytes=len(enc["data"]), sha=enc["sha"][:16])
    else:
        fq = st.file_uploader("FASTQ (.fastq or .fastq.gz)", key="fq")
        if fq is not None:
            reads_in, meta, dropped = C.parse_fastq(fq.getvalue())
            if len(reads_in) == 0:
                st.error("No reads of the expected length (372 nt) found."); reads_in = None
    if reads_in is not None and st.button("Decode", type="primary"):
        with st.spinner("Decoding..."):
            let, stt = C.decode_reads(reads_in, meta.get("bytes"), correct)
            out = C.letters_to_bytes(let, meta["bytes"]) if meta.get("bytes") else C.letters_to_bytes(let, len(let) * C.BITS // 8)
        st.session_state["dec"] = dict(out=out, stt=stt, meta=meta, dropped=dropped, correct=correct)
    dec = st.session_state.get("dec")
    if dec:
        stt, meta, out = dec["stt"], dec["meta"], dec["out"]
        st.subheader("Result")
        same = bool(meta.get("sha")) and hashlib.sha256(out).hexdigest()[:16] == meta["sha"]
        (st.success if same else st.error)("Decoded file is IDENTICAL to the original (SHA-256 matches)." if same else
                                          "Decoded file differs from the original (or no checksum in the FASTQ header).")
        a, b, c, d = st.columns(4)
        a.metric("Molecules read", f"{stt['reads_in']:,}", f"-{dec['dropped']:,} wrong length" if dec["dropped"] else None, delta_color="off")
        b.metric("Reads dropped (barcode)", f"{stt['bad_barcode']:,}", help="Barcode unreadable / not matching any strand")
        c.metric("Trimer reads ignored (not in Omega)", f"{stt['ignored_trimers']:,}" if dec["correct"] else "n/a")
        d.metric("Unreadable letters", f"{stt['unreadable_letters']:,} / {stt['letters']:,}")
        st.caption(f"Barcodes repaired: {stt['repaired_barcodes']:,} · strands with no molecule: {stt['strands_missing']:,} · "
                   f"unreadable letters are written as zeros.")
        oname = meta.get("name", "decoded.bin")
        p1, p2 = st.columns(2)
        if enc and meta.get("sha") == enc["sha"][:16]:
            with p1:
                st.subheader("Original"); preview(enc["data"], enc["name"])
        with p2 if enc and meta.get("sha") == enc["sha"][:16] else p1:
            st.subheader("Decoded"); preview(out, oname)
        st.markdown("**Decoded file metadata**"); meta_table(oname, out)
        st.download_button("Download decoded file", out, file_name="decoded_" + oname, on_click="ignore")
