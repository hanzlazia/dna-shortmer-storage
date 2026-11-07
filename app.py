"""Combinatorial shortmer DNA storage: interactive encoder / channel / decoder.

Implements the method of Preuss, Yakhini & Anavy, "Data Storage Based on
Combinatorial Synthesis of DNA Shortmers" (bioRxiv 2021.08.01.454622).
Run with:  streamlit run app.py
"""
from __future__ import annotations

import colorsys
import hashlib
from dataclasses import fields, replace
from math import ceil

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from shortmer import capacity, synthesis
from shortmer.alphabet import hamming
from shortmer.channel import inject_letter_errors, reconstruct, simulate_reads
from shortmer.codec import (ERASED, KIND_ORDER, LRS, PADDING, PADINFO, PAYLOAD, PRESETS,
                            XRS, Params, decode, encode, make_alphabet)

alt.data_transformers.disable_max_rows()
MAX_BYTES = 200_000
MAX_HEATMAP_ROWS = 160

st.set_page_config(page_title="Combinatorial Shortmer DNA Storage", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"], .stMarkdown, .stText { font-family: "Georgia","Times New Roman",serif; }
h1 { font-size: 1.7rem !important; font-weight: 600 !important; letter-spacing: 0; }
h2, h3 { font-weight: 600 !important; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
code, .mono { font-family: "Consolas","Menlo",monospace; }
.chip { display:inline-block; padding:1px 5px; margin:1px; border-radius:3px; font:12px Consolas,monospace; color:#222; }
.stack { display:inline-block; vertical-align:top; margin:0 3px 6px 0; padding:2px; border:1px solid #cfd4d9; border-radius:4px; text-align:center; }
.bc { display:inline-block; padding:2px 6px; margin-right:6px; border-radius:3px; font:13px Consolas,monospace; background:#33414f; color:#fff; }
.bcr { background:#8795a3; }
.small { color:#5b6670; font-size:0.9rem; }
</style>""", unsafe_allow_html=True)

KIND_COLORS = {PAYLOAD: "#3b5b7a", LRS: "#9db7d1", XRS: "#c9a66b", PADDING: "#c4c4c4", PADINFO: "#7a4b4b"}
STATE_COLORS = {"Correct": "#d5dde3", "Wrong": "#b23a48", "Erased": "#222222"}


# ------------------------------------------------------------------ helpers
def kmer_color(i: int) -> str:
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.84, 0.55)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def chip(kmer: str, omega: list[str]) -> str:
    return f'<span class="chip" style="background:{kmer_color(omega.index(kmer))}">{kmer}</span>'


def letter_stack(v: int, alpha, label: str = "") -> str:
    inner = "<br>".join(chip(k, alpha.omega) for k in alpha.kmers(v))
    cap = f'<div class="small mono">{label}</div>' if label else ""
    return f'<div class="stack">{cap}{inner}</div>'


def barcode_html(enc, i: int) -> str:
    nd = enc.nb * 2
    bc = enc.barcodes[i]
    return f'<span class="bc">{bc[:nd]}</span><span class="bc bcr">{bc[nd:]}</span>'


def heat(df: pd.DataFrame, color_field: str, domain: list[str], colors: dict, title: str):
    h = max(180, min(520, 3 * (df["row"].max() + 1)))
    return alt.Chart(df, title=title).mark_rect().encode(
        x=alt.X("col:O", axis=None),
        y=alt.Y("row:O", axis=alt.Axis(labels=False, ticks=False, title="sequences"), sort="ascending"),
        color=alt.Color(f"{color_field}:N", scale=alt.Scale(domain=domain, range=[colors[d] for d in domain]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["row", "col", "letter", color_field],
    ).properties(height=h)


def matrix_df(mat: list, kinds=None, truth=None) -> pd.DataFrame:
    recs = []
    for i, row in enumerate(mat[:MAX_HEATMAP_ROWS]):
        for c, v in enumerate(row):
            d = {"row": i, "col": c, "letter": v}
            if kinds is not None:
                d["category"] = kinds[i][c]
            if truth is not None:
                d["state"] = "Erased" if v == ERASED else ("Correct" if v == truth[i][c] else "Wrong")
            recs.append(d)
    return pd.DataFrame(recs)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Input")
src = st.sidebar.radio("Source", ["Upload a file (any type)", "Type text"], label_visibility="collapsed",
                       index=1 if "text" in st.query_params else 0)
if src.startswith("Upload"):
    up = st.sidebar.file_uploader("File", label_visibility="collapsed")
    if up is None:
        st.title("Combinatorial Shortmer DNA Storage")
        st.info("Upload any file in the sidebar (or choose *Type text*) to encode it into combinatorial "
                f"DNA shortmer sequences. Maximum size {MAX_BYTES // 1000} KB.")
        st.stop()
    data, fname = up.getvalue(), up.name
else:
    txt = st.sidebar.text_area("Message", "DNA is a high-density, durable archival medium. "
                               "Combinatorial shortmer synthesis encodes more than two bits per synthesis cycle.",
                               height=120)
    data, fname = txt.encode("utf-8"), "message.txt"
if not data:
    st.warning("The input is empty.")
    st.stop()
if len(data) > MAX_BYTES:
    st.error(f"File is {len(data):,} bytes; this interactive demo accepts at most {MAX_BYTES:,} bytes.")
    st.stop()

st.sidebar.header("Encoding scheme")
FIELDS = [f.name for f in fields(Params)]


def load_preset():
    if st.session_state.preset in PRESETS:
        pr = PRESETS[st.session_state.preset]
        for f in FIELDS:
            st.session_state[f"p_{f}"] = getattr(pr, f)


if "preset" not in st.session_state:
    st.session_state.preset = list(PRESETS)[0]
    load_preset()
st.sidebar.selectbox("Preset", list(PRESETS) + ["Custom"], key="preset", on_change=load_preset,
                     help="Loads parameter values; edit any value below to customise.")
LBL = {
    "N": ("N = |Ω| (k-mers)", 4, 64), "K": ("K (k-mers per letter)", 1, 63),
    "k": ("k-mer length", 2, 5), "d": ("min. Hamming distance of Ω", 1, 4),
    "payload_letters": ("payload letters / sequence", 4, 400), "rs_len": ("lengthwise RS parity letters", 0, 60),
    "block_data": ("data sequences / block", 2, 200), "block_parity": ("crosswise RS parity sequences / block", 0, 40),
    "bc_parity": ("barcode RS parity symbols (GF(16))", 2, 6),
}
for f in FIELDS:
    lab, lo, hi = LBL[f]
    st.sidebar.number_input(lab, lo, hi, key=f"p_{f}", step=1)
params = Params(**{f: int(st.session_state[f"p_{f}"]) for f in FIELDS})
seed = int(st.sidebar.number_input("Random seed", 0, 10**6, 1))
st.sidebar.markdown(
    '<p class="small">Method: I. Preuss, Z. Yakhini, L. Anavy, <i>Data Storage Based on Combinatorial '
    'Synthesis of DNA Shortmers</i>, bioRxiv 2021.08.01.454622.</p>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def run_encode(data: bytes, params: Params):
    return encode(data, params)


try:
    enc = run_encode(data, params)
except ValueError as e:
    st.title("Combinatorial Shortmer DNA Storage")
    st.error(f"Invalid parameter combination: {e}")
    st.stop()
alpha, r = enc.alphabet, enc.alphabet.bits
if alpha.min_distance < 2:
    st.sidebar.warning("Ω has minimum Hamming distance < 2: single-base substitutions cannot be detected.")

st.title("Combinatorial Shortmer DNA Storage")
st.caption(f"Encoding **{fname}** ({len(data):,} bytes) with a binomial alphabet over N = {alpha.N} "
           f"{alpha.k}-mers, K = {alpha.K}  →  |Σ| = C({alpha.N},{alpha.K}) = {alpha.size:,}; "
           f"{r} bits per letter (uses {alpha.usable:,} letters).")

t_over, t_enc, t_syn, t_dec, t_exp, t_cap = st.tabs(
    ["Workflow", "1 · Encoding", "2 · Synthesis", "3 · Channel and decoding", "Experiments", "Capacity and Ω"])

# ------------------------------------------------------------------ workflow
with t_over:
    n_data_rows = sum(k == "data" or k == "info" for k in enc.row_kind)
    mol_nt = enc.nb * 2 + params.bc_parity * 2 + params.width * alpha.k
    flow = pd.DataFrame([
        ("i", "Binary → combinatorial letters", f"{len(data)*8:,} bits → {enc.n_data_letters:,} letters of {r} bits"),
        ("ii", "Sequences + 2D Reed–Solomon",
         f"{enc.total_rows} sequences in {enc.n_blocks} block(s); RS({params.width},{params.payload_letters}) "
         f"lengthwise, RS({params.block_size},{params.block_data}) crosswise"),
        ("ii", "Barcodes", f"{enc.nb} data + {params.bc_parity} RS symbols = {(enc.nb+params.bc_parity)*2} nt per sequence"),
        ("iii", "Combinatorial synthesis", f"{params.width} letters × {alpha.K} {alpha.k}-mers; molecule length "
         f"{mol_nt} nt; {(enc.nb+params.bc_parity)*2 + params.width*(alpha.K+2):,} dispense operations per sequence"),
        ("iv", "Sequencing", "Reads of many molecules per barcode (channel tab)"),
        ("v", "Reconstruction", "Group by barcode; top-K k-mers per position give each letter"),
        ("vi", "Error correction", "Lengthwise RS, then crosswise RS (errors + erasures)"),
        ("vii", "Letters → binary", "Strip padding recorded in the last sequence"),
    ], columns=["Step", "Operation", "This input"])
    st.dataframe(flow, hide_index=True, width="stretch")
    st.markdown(
        "**Scope.** Steps i–iii are the *encoding* part (tab 1 and 2). Steps iv–vii and the error-injection "
        "controls (tab 3) let you perturb the stored data and watch each correction layer repair it, to "
        "verify the encoding end to end.")
    st.markdown(
        "**Key idea.** A standard DNA position carries 2 bits (one of A, C, G, T). Here a *letter* is a "
        f"subset of K = {alpha.K} out of N = {alpha.N} short k-mers. The synthesis mixes those K k-mers "
        f"and couples them to the growing strand in a single cycle, so one position stores log₂ C(N,K) = "
        f"{r} bits. The letter is recovered by sequencing many molecules of the same barcode and "
        "reading which K k-mers occur at the position.")

# ------------------------------------------------------------------ encoding
with t_enc:
    st.subheader("Step i · Binary message → combinatorial letters")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Message", f"{len(data):,} B")
    c2.metric("Bits per letter", r)
    c3.metric("Letters", f"{enc.n_data_letters:,}")
    c4.metric("Bit padding", f"{enc.bit_pad} b")
    st.caption(f"SHA-256 of input: `{sha(data)}`")
    n_show = st.slider("Letters to display", 4, 24, 10)
    rows = []
    for i in range(min(n_show, enc.n_data_letters)):
        chunk = enc.bits[i * r:(i + 1) * r].ljust(r, "0")
        v = enc.data_letters[i]
        rows.append({"position": i, "bits": chunk, "letter σ (index)": v,
                     "subset S ⊂ Ω": "{" + ", ".join(f"X{j+1}" for j in alpha.subset(v)) + "}",
                     "k-mers": ", ".join(alpha.kmers(v))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Letters are ranked lexicographically among the K-subsets of Ω (σ₀ = {X₁,…,X_K}); the paper's "
               "σ numbering is arbitrary. Indices ≥ 2^r are unused.")
    with st.expander("Ω: the k-mer building blocks"):
        st.markdown(" ".join(f'<span class="mono">X{i+1}</span>&nbsp;{chip(k, alpha.omega)}'
                             for i, k in enumerate(alpha.omega)), unsafe_allow_html=True)
        st.caption(f"Minimum Hamming distance of Ω is {alpha.min_distance}; a single nucleotide substitution "
                   "therefore turns a k-mer of Ω into a word outside Ω, which is detected and ignored."
                   if alpha.min_distance >= 2 else "Distance < 2: substitutions may map onto other members of Ω.")

    st.subheader("Step ii · Sequences, 2D Reed–Solomon and padding")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sequences", enc.total_rows)
    c2.metric("Blocks", enc.n_blocks)
    c3.metric("Total padding", f"{enc.pad_bits_total} b")
    c4.metric("Code rate", f"{len(data)*8 / (enc.total_rows*params.width*r):.1%}")
    st.markdown(
        f"Each sequence holds {params.payload_letters} payload letters protected lengthwise by "
        f"RS({params.width},{params.payload_letters}) over GF(2^{r}). Every block of {params.block_data} data "
        f"sequences is protected columnwise by {params.block_parity} parity sequences, so whole sequences can be "
        "lost. The last data sequence stores the total padding length (letters, little-endian), as in the "
        "paper's Figure 6.")
    if enc.total_rows > MAX_HEATMAP_ROWS:
        st.caption(f"Showing the first {MAX_HEATMAP_ROWS} of {enc.total_rows} sequences.")
    st.altair_chart(heat(matrix_df(enc.matrix, kinds=enc.kinds), "category", KIND_ORDER, KIND_COLORS,
                         "Layout of the encoded letter matrix"), width="stretch")
    with st.expander("Raw letter matrix"):
        st.dataframe(pd.DataFrame(enc.matrix[:300]), width="stretch")

    st.subheader("Step ii · Barcodes")
    st.markdown(f"The sequence index is written as {enc.nb} symbols over GF(16) (4 bits = 2 nt each) and "
                f"protected by {params.bc_parity} RS parity symbols, which repairs one corrupted symbol so "
                "that reads can still be grouped.")
    st.markdown("".join(f'<div style="margin:2px 0"><span class="mono small">seq {i:>4}</span> '
                        f'{barcode_html(enc, i)}</div>' for i in range(min(8, enc.total_rows))),
                unsafe_allow_html=True)

    st.subheader("Step iii · Oligo design")
    sel = st.number_input("Sequence", 0, enc.total_rows - 1, 0, key="sel_enc")
    st.markdown(f'**Sequence {sel}** — region: `{enc.row_kind[sel]}`'.replace("xrs", "crosswise parity"))
    shown = min(params.width, 16)
    html = barcode_html(enc, sel) + "".join(
        letter_stack(v, alpha, f"{c}") for c, v in enumerate(enc.matrix[sel][:shown]))
    if params.width > shown:
        html += '<span class="small"> … ' + f"{params.width - shown} more letters</span>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption("Each column is one synthesis position: the k-mers listed are mixed and coupled together. "
               f"The oligo is {enc.nb*2 + params.bc_parity*2 + params.width*alpha.k} nt long.")
    design = pd.DataFrame({
        "sequence": range(enc.total_rows), "barcode": enc.barcodes, "role": enc.row_kind,
        "letters": [" ".join(map(str, row)) for row in enc.matrix],
        "kmer_sets": [" ".join("{" + "|".join(alpha.kmers(v)) + "}" for v in row) for row in enc.matrix]})
    st.download_button("Download design (CSV)", design.to_csv(index=False).encode(),
                       f"{fname}.design.csv", "text/csv")

# ------------------------------------------------------------------ synthesis
with t_syn:
    st.subheader("Combinatorial shortmer synthesis (paper Figure 2)")
    sq = st.number_input("Sequence", 0, enc.total_rows - 1, 0, key="sel_syn")
    nlet = st.slider("Letters to synthesise in this walk-through", 1, params.width, min(3, params.width))
    steps = synthesis.protocol(enc.barcodes[sq], enc.matrix[sq][:nlet], alpha)
    s_idx = st.slider("Step", 1, len(steps), min(len(steps), 1 + len(enc.barcodes[sq]) + alpha.K + 2))
    s = steps[s_idx - 1]

    def colored(strand: str) -> str:
        out, i = "", 0
        while i < len(strand):
            if strand[i] == "{":
                j = strand.index("}", i)
                out += '<span class="stack">' + "<br>".join(chip(k, alpha.omega) for k in strand[i+1:j].split(",")) + "</span>"
                i = j + 1
            else:
                out += f'<span class="bc">{strand[i]}</span>'
                i += 1
        return out

    st.markdown(f"**Step {s.n} / {len(steps)} · {s.phase}** — {s.action}")
    c1, c2 = st.columns([1, 2])
    c1.markdown("**Mixing chamber**")
    c1.markdown(" ".join(chip(k, alpha.omega) for k in s.chamber) or "<span class='small'>empty</span>", unsafe_allow_html=True)
    c2.markdown("**Strand on the surface (consensus)**")
    c2.markdown(colored(s.strand) or "<span class='small'>linker only</span>", unsafe_allow_html=True)
    with st.expander("Full protocol for this walk-through"):
        st.dataframe(pd.DataFrame([{"step": x.n, "phase": x.phase, "action": x.action,
                                    "mixing chamber": ", ".join(x.chamber)} for x in steps]),
                     hide_index=True, width="stretch")
    cc = synthesis.cycle_counts(len(enc.barcodes[sq]), params.width, alpha.K)
    std_cycles = ceil(params.width * r / 2)
    st.markdown("**Operations per full-length sequence**")
    st.dataframe(pd.DataFrame({"operation": list(cc), "count": list(cc.values())}), hide_index=True)
    st.caption(f"The payload ({params.width} letters = {params.width * r} bits) would need {std_cycles} "
               f"elongation cycles in standard DNA synthesis (2 bits per cycle) but needs {params.width} cycles here, "
               f"at the price of {alpha.K} extra k-mer injections per cycle and one mixing step. "
               "Hardware needs N extra cartridges plus a mixing chamber.")

# ------------------------------------------------------------------ channel + decode
with t_dec:
    st.subheader("Introduce errors, then decode")
    mode = st.radio("Error model", ["Direct letter-level injection (explanatory)",
                                    "Molecular simulation: synthesis + sequencing (paper, Sec. 3.1)"],
                    horizontal=True)
    rng = np.random.default_rng(seed)
    truth = enc.matrix
    log: list[str] = []
    ch_info: dict = {}
    if mode.startswith("Direct"):
        c1, c2, c3 = st.columns(3)
        n_rand = c1.slider("Random letter errors", 0, max(1, enc.total_rows * params.width // 4), 0,
                           help="Substitutions at uniformly random cells of the whole matrix.")
        burst_row = c2.number_input("Burst: sequence index", 0, enc.total_rows - 1, 0)
        burst_len = c2.slider("Burst: corrupted letters in that sequence", 0, params.width, 0,
                              help=f"Lengthwise RS repairs up to {params.rs_len // 2} per sequence; beyond that "
                                   "the crosswise code must take over.")
        n_lose = c3.slider("Lost sequences (dropout)", 0, enc.total_rows, 0,
                           help=f"Crosswise RS repairs up to {params.block_parity} lost sequences per block.")
        st.caption("Optional: edit specific letters (sequence, position, new letter).")
        tgt = st.data_editor(pd.DataFrame({"sequence": pd.Series(dtype=int), "position": pd.Series(dtype=int),
                                           "new_letter": pd.Series(dtype=int)}),
                             num_rows="dynamic", width="stretch", key="targets")
        targeted = [(int(a), int(b), int(c)) for a, b, c in tgt.dropna().itertuples(index=False)]
        inj = inject_letter_errors(enc, rng, random_errors=n_rand, burst_row=int(burst_row), burst_len=burst_len,
                                   lose_rows=n_lose, targeted=targeted)
        received, log = inj.received, inj.log
    else:
        c1, c2, c3, c4 = st.columns(4)
        depth = c1.slider("Reads per barcode (mean)", 5, 500, 100)
        p_sub = c2.select_slider("Substitution rate / nt", [0, 1e-4, 1e-3, 5e-3, 1e-2, 3e-2], 1e-3, format_func=lambda x: f"{x:g}")
        p_ins = c3.select_slider("Insertion rate / k-mer", [0, 1e-4, 1e-3, 5e-3, 1e-2], 0, format_func=lambda x: f"{x:g}")
        p_del = c4.select_slider("Deletion rate / k-mer", [0, 1e-4, 1e-3, 5e-3, 1e-2], 0, format_func=lambda x: f"{x:g}")
        n_lose = st.slider("Sequences lost before sequencing", 0, enc.total_rows, 0)
        reads, st_ch = simulate_reads(enc, rng, depth=depth, p_sub=p_sub, p_ins=p_ins, p_del=p_del, lose_rows=n_lose)
        received, st_ch = reconstruct(reads, params, alpha, st_ch, enc.total_rows)
        if len(received) != enc.total_rows:
            received = (received + [None] * enc.total_rows)[:enc.total_rows]
        ch_info = {"reads": st_ch.reads, "reads discarded (barcode)": st_ch.discarded_barcode,
                   "sub events": st_ch.sub_events, "k-mer insertions": st_ch.ins_events,
                   "k-mer deletions": st_ch.del_events, "barcodes recovered": st_ch.groups,
                   "mean reads / barcode": round(st_ch.mean_depth, 1)}

    rep = decode(received, params, alpha)
    ok = rep.ok and rep.recovered == data
    if ok:
        st.success(f"Perfect recovery. SHA-256 of the recovered file matches the input "
                   f"(`{sha(rep.recovered)[:16]}…`).")
    elif rep.ok:
        bad = sum(a != b for a, b in zip(rep.recovered, data)) + abs(len(rep.recovered) - len(data))
        st.error(f"Decoder returned data that differs from the input ({bad:,} byte differences): "
                 "error correction was exceeded and the corruption was not detected.")
    else:
        st.error(f"Decoding failed: {rep.message}. The injected damage exceeds the capacity of the error correction.")

    if mode.startswith("Direct") and log:
        with st.expander(f"Injection log ({len(log)} events)"):
            st.text("\n".join(log[:500]))
    if ch_info:
        st.dataframe(pd.DataFrame([ch_info]), hide_index=True, width="stretch")

    stages = rep.stage_matrices
    names = list(stages)
    summary = []
    for nm in names:
        m = stages[nm]
        wrong = sum(1 for i, row in enumerate(m) for c, v in enumerate(row) if v != ERASED and v != truth[i][c])
        er = sum(1 for row in m for v in row if v == ERASED)
        summary.append({"stage": nm, "wrong letters": wrong, "erased letters": er,
                        "wrong or erased (%)": round(100 * (wrong + er) / (len(m) * params.width), 3)})
    st.markdown("**Letters that differ from the encoded data after each stage**")
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    s = rep.stats
    st.caption(f"Missing sequences: {s['rows_missing']} · lengthwise RS failed on {s['lengthwise_failed']} sequences "
               f"(fixed {s['lengthwise_symbols_fixed']} letters) · crosswise RS fixed {s['crosswise_symbols_fixed']} "
               f"letters, {s['crosswise_failed_columns']} columns failed.")
    tabs = st.tabs(names)
    for tb, nm in zip(tabs, names):
        with tb:
            st.altair_chart(heat(matrix_df(stages[nm], truth=truth), "state", list(STATE_COLORS), STATE_COLORS,
                                 f"Letter state: {nm}"), width="stretch")
    if rep.ok and rep.recovered is not None:
        st.download_button("Download recovered file", rep.recovered, f"recovered_{fname}")

# ------------------------------------------------------------------ experiments
with t_exp:
    st.subheader("Reconstruction rate versus error rate and sequencing depth")
    st.markdown("Repeats the full molecular pipeline (encode → synthesis/sequencing simulation → reconstruction → "
                "2D RS decoding), analogous to Figure 4 of the paper. Uses at most the first 3 KB of the input.")
    c1, c2, c3 = st.columns(3)
    etype = c1.selectbox("Error type", ["Substitution", "Insertion", "Deletion"])
    depths = c2.multiselect("Reads per barcode", [10, 20, 50, 100, 200, 500], [20, 50, 100, 200])
    trials = c3.slider("Trials per setting", 1, 10, 3)
    rates = st.multiselect("Error rates", [0, 1e-4, 1e-3, 1e-2], [0, 1e-4, 1e-3, 1e-2], format_func=lambda x: f"{x:g}")
    if st.button("Run experiment") and depths and rates:
        sub = data[:3000]
        e2 = encode(sub, params)
        out, prog = [], st.progress(0.0)
        total_runs, done = len(depths) * len(rates) * trials, 0
        for dpt in depths:
            for rt in rates:
                good = 0
                for t in range(trials):
                    g = np.random.default_rng(seed * 1000 + t)
                    kw = dict(p_sub=0.0, p_ins=0.0, p_del=0.0)
                    kw[{"Substitution": "p_sub", "Insertion": "p_ins", "Deletion": "p_del"}[etype]] = rt
                    rd, sc = simulate_reads(e2, g, depth=dpt, **kw)
                    mat, sc = reconstruct(rd, params, alpha, sc, e2.total_rows)
                    mat = (mat + [None] * e2.total_rows)[:e2.total_rows]
                    good += decode(mat, params, alpha).recovered == sub
                    done += 1
                    prog.progress(done / total_runs)
                out.append({"reads": str(dpt), "error rate": rt, "full recovery (%)": 100 * good / trials})
        prog.empty()
        df = pd.DataFrame(out)
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("error rate:O", title=f"{etype.lower()} rate"), xOffset="reads:N",
            y=alt.Y("full recovery (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("reads:N", sort=[str(d) for d in sorted(depths)], title="reads / barcode",
                            scale=alt.Scale(scheme="blues")),
            tooltip=list(df.columns)).properties(height=320), width="stretch")
        st.dataframe(df.pivot(index="reads", columns="error rate", values="full recovery (%)"), width="stretch")

# ------------------------------------------------------------------ capacity
with t_cap:
    st.subheader("Logical density (paper Table 3)")
    st.markdown("Recomputed from the formulas for a 1 GB message with 120 payload letters, 14 RS letters, a 16 nt "
                "barcode and 42 + 6 sequence blocks; matches the published table.")
    st.dataframe(capacity.table3(), hide_index=True, width="stretch")
    st.markdown("**Current configuration**")
    cur = capacity.capacity_row("Binomial", params.N, params.K, payload_letters=params.payload_letters,
                                rs_len=params.rs_len, barcode_nt=(enc.nb + params.bc_parity) * 2,
                                block_data=params.block_data, block_parity=params.block_parity,
                                size_bytes=len(data))
    st.dataframe(pd.DataFrame([cur]), hide_index=True, width="stretch")
    st.subheader("Hamming distances within Ω")
    dm = pd.DataFrame([{"a": a, "b": b, "distance": hamming(a, b)} for a in alpha.omega for b in alpha.omega])
    st.altair_chart(alt.Chart(dm).mark_rect().encode(
        x=alt.X("a:N", sort=alpha.omega, title=None), y=alt.Y("b:N", sort=alpha.omega, title=None),
        color=alt.Color("distance:Q", scale=alt.Scale(scheme="blues")), tooltip=["a", "b", "distance"]
    ).properties(height=380, width=380))
