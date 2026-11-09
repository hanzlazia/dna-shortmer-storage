"""Encode a file into combinatorial shortmer DNA (Preuss, Yakhini & Anavy 2021), download it, edit it, decode it."""
import hashlib
import io
import lzma
import math
import mimetypes
import numpy as np
import pandas as pd
import streamlit as st
import shortmer_core as C

st.set_page_config(page_title="Shortmer DNA Storage", layout="wide")
if getattr(C, "VERSION", 0) < 9:
    st.error("shortmer_core.py is outdated. Upload the shortmer_core.py from the latest zip (and delete dna_codec.py / codec.py).")
    st.stop()
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


def sniff(data: bytes, name: str) -> str:
    """File type from the content (so a renamed or .bin file still previews); falls back to the file name."""
    h = data[:16]
    if h[:3] == b"ID3" or (len(h) > 1 and h[0] == 0xFF and h[1] & 0xE0 == 0xE0): return "audio/mpeg"
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE": return "audio/wav"
    if h[:4] == b"OggS": return "audio/ogg"
    if h[:4] == b"fLaC": return "audio/flac"
    if h[:3] == b"\xff\xd8\xff": return "image/jpeg"
    if h[:8] == b"\x89PNG\r\n\x1a\n": return "image/png"
    if h[:3] == b"GIF": return "image/gif"
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP": return "image/webp"
    if h[4:8] == b"ftyp": return "audio/mp4" if h[8:12] in (b"M4A ", b"M4B ") else "video/mp4"
    if h[:4] == b"\x1aE\xdf\xa3": return "video/webm"
    if h[:4] == b"%PDF": return "application/pdf"
    return mimetypes.guess_type(name)[0] or ""


def preview(data: bytes, name: str):
    mt = sniff(data, name)
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
                               "Value": [name, sniff(data, name) or "unknown", f"{len(data):,} bytes",
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


def shrink_av(data: bytes, name: str, kind: str, a: int, b: int) -> bytes:
    """audio: a = kbit/s;  video: a = width in pixels, b = CRF (higher = smaller). Needs ffmpeg."""
    import os, shutil, subprocess, tempfile
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not available")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in" + (os.path.splitext(name)[1] or ".bin"))
        open(src, "wb").write(data)
        if kind == "audio":
            out = os.path.join(d, "out.mp3")
            cmd = [ff, "-y", "-i", src, "-vn", "-ac", "1", "-ar", "22050", "-b:a", f"{a}k", out]
        else:
            out = os.path.join(d, "out.mp4")
            cmd = [ff, "-y", "-i", src, "-vf", f"scale='min({a},iw)':-2", "-c:v", "libx264", "-crf", str(b), "-preset", "veryfast",
                   "-c:a", "aac", "-b:a", "32k", "-ac", "1", "-movflags", "+faststart", out]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return open(out, "rb").read()


def safe_limit(cap: int):
    """Largest file (bytes) that this app can store AND read back exactly, and the copies needed for it."""
    best = (0, 0)
    for c in range(10, 301):
        n_data = (cap // c) * C.RS_K // (C.RS_K + C.RS_P)
        if C.expected_lost(c, n_data, C.RS_DEFAULT) < 0.1 and n_data * 180 > best[0]:
            best = (n_data * 180, c)
    return best


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


def diff_view(clean, noisy, copies, sidx, nmol):
    """Molecules of one strand that were hit by errors: where it happened, the original and the changed piece."""
    c, n = clean[sidx * copies:(sidx + 1) * copies], noisy[sidx * copies:(sidx + 1) * copies]
    hit = np.flatnonzero((c != n).any(axis=1))
    if len(hit) == 0:
        st.info("No molecule of this strand was hit by an error. Try another strand number, or raise the error rate.")
        return
    st.caption(f"{len(hit)} of {copies} molecules of strand {sidx} contain at least one error.")
    red = "background:#f4b6b6;border-radius:4px;padding:1px 5px;font-family:monospace;font-size:13px"
    rows = []
    for r in hit[:nmol]:
        bc = [i for i in np.flatnonzero(c[r, :C.BARCODE_LEN] != n[r, :C.BARCODE_LEN])]
        tri = sorted({(i - C.BARCODE_LEN) // 3 for i in np.flatnonzero(c[r] != n[r]) if i >= C.BARCODE_LEN})
        if bc:
            b0, b1 = c[r, :C.BARCODE_LEN].tobytes().decode(), n[r, :C.BARCODE_LEN].tobytes().decode()
            rows.append((r, "barcode", f"<code>{b0}</code>", f"<code>{b1}</code>"))
        for p in tri:
            t0 = c[r, C.BARCODE_LEN + 3 * p:C.BARCODE_LEN + 3 * p + 3].tobytes().decode()
            t1 = n[r, C.BARCODE_LEN + 3 * p:C.BARCODE_LEN + 3 * p + 3].tobytes().decode()
            rows.append((r, f"position {p}", chip(t0), chip(t1) if t1 in C.OMEGA else f"<span style='{red}'>{t1}</span>"))
    H = ["<table style='border-collapse:collapse;font-size:14px'><tr><th style='text-align:left;padding:2px 14px'>Molecule</th>"
         "<th style='text-align:left;padding:2px 14px'>Where</th><th style='text-align:left;padding:2px 14px'>Original</th>"
         "<th style='text-align:left;padding:2px 14px'>After the error</th></tr>"]
    for r, where, o, new in rows:
        H.append(f"<tr style='border-top:1px solid #e3e8ed'><td style='padding:3px 14px'>{r}</td><td style='padding:3px 14px'>{where}</td>"
                 f"<td style='padding:3px 14px'>{o}</td><td style='padding:3px 14px'>{new}</td></tr>")
    st.markdown("".join(H) + "</table>", unsafe_allow_html=True)
    st.caption("Red = the changed trimer is no longer one of the 16 trimers of Omega, so it is recognised as an error.")


st.markdown("""<style>
.stButton button, .stDownloadButton button {border-radius:10px;padding:.55rem 1.1rem;font-weight:600;border:1px solid #cdd5dd;background:#fff;color:#33414f;
  box-shadow:0 1px 2px rgba(20,30,40,.08);transition:all .15s ease}
.stButton button:hover, .stDownloadButton button:hover {border-color:#3d5a80;color:#3d5a80;box-shadow:0 6px 14px rgba(61,90,128,.20);transform:translateY(-1px)}
.stButton button[data-testid="stBaseButton-primary"] {background:linear-gradient(135deg,#33414f,#3d5a80);color:#fff;border:none;padding:.75rem 2rem;font-size:1.05rem}
.stButton button[data-testid="stBaseButton-primary"]:hover {color:#fff;filter:brightness(1.12)}
div[data-testid="stMetric"] {background:#f3f5f7;border:1px solid #e3e8ed;border-radius:12px;padding:.7rem 1rem}
h2 {border-left:5px solid #3d5a80;padding-left:.7rem}
</style>""", unsafe_allow_html=True)


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
kind = sniff(orig, oname).split("/")[0]
modes = ["No compression", "Lossless compression (LZMA) - exact recovery. Does NOT help files that are already compressed (JPEG, MP3, MP4, ZIP)"]
if kind in ("image", "audio", "video"):
    modes.append(f"Shrink the {kind} (lossy) - real size reduction, quality is reduced")
too_big_file = len(orig) > safe_limit(MAX_READS)[0]
mode = st.radio("Compression", modes, label_visibility="collapsed", index=len(modes) - 1 if (too_big_file and len(modes) == 3) else 0,
                key=f"mode_{sha16(orig)}")
payload, pname, comp = orig, oname, "none"
if mode.startswith("Lossless"):
    payload, comp = lzma.compress(orig, preset=9 | lzma.PRESET_EXTREME), "lzma"
elif mode.startswith("Shrink"):
    q1, q2 = st.columns(2)
    stem = oname.rsplit(".", 1)[0]
    try:
        if kind == "image":
            side = q1.slider("Longest side (pixels)", 64, 3000, 640); qual_j = q2.slider("JPEG quality", 5, 95, 40)
            payload, pname = shrink_image(orig, side, qual_j), stem + "_small.jpg"
        elif kind == "audio":
            payload, pname = shrink_av(orig, oname, "audio", 32, 0), stem + "_small.mp3"
            st.caption("Shrunk to MP3 · mono · 22 kHz · 32 kbit/s")
        else:
            width = q1.slider("Video width (pixels)", 96, 1280, 240); crf = q2.slider("Quality loss (CRF, higher = smaller)", 18, 45, 36)
            payload, pname = shrink_av(orig, oname, "video", width, crf), stem + "_small.mp4"
    except Exception as e:
        st.error(f"Could not shrink this {kind} ({type(e).__name__}). Choose another option."); payload = orig
psha = sha16(payload)
k1, k2, k3 = st.columns(3)
k1.metric("Original", f"{len(orig):,} bytes")
k2.metric("To be stored in DNA", f"{len(payload):,} bytes", f"{(len(payload) / len(orig) - 1):+.0%}" if len(payload) != len(orig) else None, delta_color="off")
k3.metric("Bits", f"{len(payload) * 8:,}")
if mode.startswith("Shrink"):
    st.markdown("**Shrunk file (this is what will be stored in DNA)**"); preview(payload, pname)
elif comp == "lzma":
    st.caption("LZMA output is just compressed bytes, not a playable/viewable file, so there is nothing to preview here. "
               "After decoding it is decompressed and you will see the original again in Step 6." +
               (" Almost no gain here: this file type is already compressed." if len(payload) > 0.9 * len(orig) else ""))
cap_bytes, cap_copies = safe_limit(MAX_READS)
if len(payload) > cap_bytes:
    st.warning(f"This app can store and exactly recover files up to about **{cap_bytes // 1000} KB** (with Reed–Solomon, ≈ {cap_copies} copies per strand). "
               f"The file to be stored is {len(payload) // 1000} KB. Use a lossy shrink option above to get below that.")
b1, b2 = st.columns(2)
if payload != orig:
    b1.download_button("Download compressed file", payload, file_name=pname if comp == "none" else pname + ".lzma", on_click="ignore", icon=":material/download:")
if len(payload) <= 3_000_000:
    b2.download_button("Download bits (0/1 text)", lambda: "".join(map(str, np.unpackbits(np.frombuffer(payload, np.uint8)))).encode(),
                       file_name=pname + "_bits.txt", on_click="ignore", icon=":material/download:")

letters_data = C.bytes_to_letters(payload)
n_data = len(letters_data) // C.LPS
m1, m2, m3 = st.columns(3)
m1.metric("Letters (12 bits each)", f"{len(letters_data):,}"); m2.metric("Data strands (120 letters)", f"{n_data:,}")
m3.metric("Bits per nucleotide", f"{C.BITS / 3:.0f}", help="Standard DNA: 2")
with st.expander("The encoding, letter by letter (first letters of the file)", expanded=True):
    bits = np.unpackbits(np.frombuffer(payload[:10], dtype=np.uint8))
    rows = [{"Position": i, "12 bits": "".join(map(str, np.pad(bits, (0, 96))[12 * i:12 * i + 12])), "Letter number": int(letters_data[i]),
             "Letter = 5 trimers": " ".join(C.letter_trimers(int(letters_data[i]))),
             "As Omega indices": " ".join(f"X{j + 1}" for j in C.SUBSETS[letters_data[i]])} for i in range(6)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.markdown("Omega (paper Table 1): " + " ".join(f"<small>X{i + 1}</small>{chip(t)}" for i, t in enumerate(C.OMEGA)), unsafe_allow_html=True)

# ---------------------------------------------------------------- 3 synthesis
step(3, "Synthesis: make the molecules")
rs_choice = st.radio("Error protection across strands", ["With Reed–Solomon (the paper's method)", "Without Reed–Solomon"], horizontal=True,
                     help="With: extra parity strands are added to every group of data strands. Letters that could not be read are rebuilt, so far fewer copies "
                          "are needed. Without: every letter must be read directly, so many more copies are needed.")
rs_cfg = C.RS_DEFAULT if rs_choice.startswith("With ") else None
if rs_cfg:
    st.caption(f"As in the paper: Reed–Solomon over GF(2¹²), {rs_cfg[1]} parity strands for every {rs_cfg[0]} data strands (+{rs_cfg[1] / rs_cfg[0]:.0%} strands). "
               f"In each group, any {rs_cfg[1]} unreadable letters at the same position are rebuilt automatically when decoding.")
s1, s2 = st.columns(2)
need = next((c for c in range(5, 301) if C.expected_lost(c, n_data, rs_cfg) < 0.05), 300)
copies = s1.slider("Copies (molecules) per strand", 1, 300, need, key=f"copies_{psha}_{rs_cfg}",
                   help="Each letter is a mixture of 5 trimers, but each molecule carries only ONE of them at every position, so several copies are needed to see all 5.")
seed = s2.number_input("Random seed", 0, 10_000, 1,
                       help="The molecules pick their trimers at random. The seed fixes that randomness: the same seed gives exactly the same molecules and errors every time; a different seed gives a different random outcome.")
letters = C.rs_encode(letters_data, rs_cfg) if rs_cfg else letters_data
n_letters, n_strands = len(letters), len(letters) // C.LPS
n_reads = n_strands * copies
miss = C.miss_probability(copies)
lost = C.expected_lost(copies, n_data, rs_cfg)
st.caption(f"**{n_strands:,}** strands ({n_data:,} data" + (f" + {n_strands - n_data:,} parity" if rs_cfg else "") + f") × {copies} copies = **{n_reads:,}** molecules.  "
           f"A letter misses at least one of its 5 trimers with probability **{miss:.2%}** → about **{miss * n_letters:,.0f}** unreadable letters before repair."
           + (f"  After Reed–Solomon repair, about **{lost:,.1f}** letter positions are expected to stay lost." if rs_cfg else ""))
if lost >= 0.5:
    st.warning(f"With {copies} copies about {lost:,.0f} letters will stay unreadable, so the recovered file will be damaged "
               f"(an audio/video/image may not play). Use more copies" + ("" if rs_cfg else ", turn on Reed–Solomon") + " or a smaller file (Step 2).")
if n_reads > MAX_READS:
    st.error(f"{n_reads:,} molecules is more than this app can handle ({MAX_READS:,}). Use Step 2 (shrink) or fewer copies.")
    st.stop()
ckey = f"{psha}{rs_cfg}"
clean = get_clean(ckey, letters, copies, int(seed))
v1, v2, v3 = st.columns(3)
sidx = v1.number_input("Strand number", 0, n_strands - 1, 0)
npos = v2.slider("Positions to show", 3, 16, 8)
nmol = v3.slider("Molecules to show", 1, 15, min(8, copies))
st.markdown("**Top row:** the letter at each position = its 5 trimers (the mixture that is synthesized). "
            "**Rows below:** real molecules; each one holds a single trimer of the 5 at every position.")
viewer(clean, letters, copies, sidx, npos, nmol)

# ---------------------------------------------------------------- 4 errors
step(4, "Introduce substitution errors")
p_sub = st.slider("Substitution error per nucleotide (%)", 0.0, 5.0, 0.1, 0.01, format="%.2f",
                  help="Each base of each molecule is replaced by a different base with this probability.") / 100
reads, n_changed, qual = get_noisy(ckey, copies, int(seed), p_sub, clean)
st.markdown(f"**Strand {sidx} (chosen in Step 3): the molecules that were hit by an error, before and after**")
diff_view(clean, reads, copies, sidx, nmol)
meta = dict(name=pname if comp == "none" else oname, bytes=len(payload), sha=psha, comp=comp, rsk=rs_cfg[0] if rs_cfg else 0, rsp=rs_cfg[1] if rs_cfg else 0)
if comp == "lzma":
    meta["orig"] = sha16(orig)
d1, d2, d3 = st.columns(3)
mb = len(reads) * (2 * C.READ_LEN + 140) / 1e6
d1.download_button(f"Download FASTQ ({mb:,.0f} MB)", lambda: C.to_fastq(reads, copies, meta), file_name="encoded.fastq", mime="text/plain",
                   on_click="ignore", icon=":material/download:", help="The molecules (with these errors). Edit it by hand if you like, then upload it in Step 5.")
d2.download_button("Download CSV: molecules", lambda: C.to_molecule_csv(reads, copies), file_name="molecules.csv", mime="text/csv",
                   on_click="ignore", icon=":material/table:", help="One row per molecule, trimers separated by spaces.")
if n_letters <= 300_000:
    d3.download_button("Download CSV: encoding table", lambda: C.to_letter_csv(letters), file_name="encoding_table.csv", mime="text/csv",
                       on_click="ignore", icon=":material/table:")

# ---------------------------------------------------------------- 5 decode
step(5, "Decode")
src = st.radio("Molecules to decode", ["Use the molecules from Step 4 (no upload needed)", "Upload a FASTQ file (for example the downloaded one, edited by hand)"])
reads_in, dqual, dmeta, dropped, fq = None, None, {}, 0, None
if src.startswith("Use"):
    reads_in, dqual, dmeta = reads, qual, meta
else:
    fq = st.file_uploader("FASTQ (.fastq or .fastq.gz)", key="fq")
    if fq is not None:
        reads_in, dqual, dmeta, dropped = C.parse_fastq(fq.getvalue())
        if len(reads_in) == 0:
            st.error("No reads of the expected length (372 nt) found."); reads_in = None
rs_dec = (int(dmeta.get("rsk", 0)), int(dmeta.get("rsp", 0))) if int(dmeta.get("rsk", 0) or 0) > 0 else None
if rs_dec is None:
    st.caption("Decoding: the trimers seen at every position give the letter. No Reed–Solomon was used for this data.")
else:
    st.caption(f"Decoding: the trimers seen at every position give the letter, then Reed–Solomon RS({rs_dec[0] + rs_dec[1]}, {rs_dec[0]}) rebuilds letters that could not be read.")
key = (psha, rs_cfg, copies, int(seed), p_sub, src[:3], getattr(fq, "file_id", None))
if reads_in is not None and st.button("Decode", type="primary", icon=":material/play_arrow:"):
    with st.spinner("Decoding..."):
        let, stt = C.decode_reads(reads_in, dmeta.get("bytes"), rs_dec)
        nb = dmeta.get("bytes") or len(let) * C.BITS // 8
        st.session_state["dec"] = dict(out=C.letters_to_bytes(let, nb), stt=stt, meta=dmeta, dropped=dropped, rs=rs_dec, key=key)
dec = st.session_state.get("dec")
if dec and dec["key"] == key:
    stt, dm, out = dec["stt"], dec["meta"], dec["out"]
    same = bool(dm.get("sha")) and sha16(out) == dm["sha"]
    (st.success if same else st.error)("Recovered bytes are IDENTICAL to what was written (SHA-256 matches)." if same else
                                      "Recovered bytes differ from what was written (or the FASTQ header has no checksum).")
    a, b, c, d = st.columns(4)
    a.metric("Molecules read", f"{stt['reads_in']:,}", f"-{dec['dropped']:,} wrong length" if dec["dropped"] else None, delta_color="off")
    b.metric("Unreadable letters (before repair)", f"{stt['unreadable_letters']:,} / {stt['letters']:,}")
    c.metric("Rebuilt by Reed–Solomon", f"{stt['rs_recovered']:,}" if dec["rs"] else "n/a")
    d.metric("Still unreadable (final)", f"{stt['still_unreadable']:,}")
    st.caption("Unreadable letters are written as zeros.")

    # ------------------------------------------------------------ 6 compare
    step(6, "Compare: original, compressed, recovered")
    dname = dm.get("name", "decoded.bin")
    final, fname = out, dname
    panels = []
    if dm.get("sha") == psha:
        panels.append(("Original", orig, oname, None))
        if payload != orig:
            panels.append(("Compressed (stored in DNA)", payload, pname if comp == "none" else pname + ".lzma",
                           "Compressed bytes, not viewable." if comp == "lzma" else None))
    decoded_note = None
    if dm.get("comp") == "lzma":
        decoded_name = dname + ".lzma"; decoded_note = "Compressed bytes, not viewable."
    else:
        decoded_name = dname
    panels.append(("Decoded from DNA", out, decoded_name, decoded_note))
    if dm.get("comp") == "lzma":
        try:
            final, fname = lzma.decompress(out), dname
            panels.append(("Decompressed", final, fname, None))
            st.success(f"Decompressed: {len(out):,} → {len(final):,} bytes" +
                       (" (matches the original checksum)" if dm.get("orig") == sha16(final) else " (checksum does NOT match the original)"))
        except Exception:
            st.error("The compressed data is damaged and cannot be decompressed (too many unreadable letters).")
    if not same:
        st.warning("The recovered file is damaged, so a player may refuse to open it. Fix: more copies, Reed–Solomon, fewer errors, or a smaller file.")
    cols = st.columns(len(panels))
    for col, (title, data_, name_, note) in zip(cols, panels):
        with col:
            st.subheader(title)
            if note:
                st.info(note)
            else:
                preview(data_, name_)
            meta_table(name_, data_)
    if dm.get("comp") == "none" and mode.startswith("Shrink") and dm.get("sha") == psha:
        st.caption("Lossy shrinking cannot be undone, so the decoded file is the shrunk one: there is no decompress step to get the original quality back.")
    st.download_button("Download recovered file", final, file_name="recovered_" + fname, on_click="ignore", icon=":material/download:")
