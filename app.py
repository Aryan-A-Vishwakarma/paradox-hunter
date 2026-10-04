"""Paradox Hunter - Streamlit dashboard.   Run with:  streamlit run app.py"""
import altair as alt
import pandas as pd
import streamlit as st

from paradox_hunter import datasets, findings_to_frame, make_report, scan
from paradox_hunter.viz import overall_rates, strata_long, strata_table

st.set_page_config(page_title="Paradox Hunter", page_icon="🔎", layout="wide")

EVIDENCE_BADGE = {"confirmed": ":green[CONFIRMED]", "suggestive": ":orange[SUGGESTIVE]",
                  "exploratory": ":gray[EXPLORATORY]"}
# Per-sample defaults: (target, lowest evidence tier to show)
SAMPLE_DEFAULTS = {"UC Berkeley admissions (1973)": "suggestive",
                   "Kidney stone treatment (1986)": "exploratory",
                   "Synthetic hospital (planted paradox + noise)": "confirmed"}


# ---- small compatibility helpers (Streamlit renamed width args across versions) ----------
def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner="Scanning every feature pair...")
def run_scan(df, target, positive, alpha, min_stratum, min_effect, min_evidence):
    return scan(df, target, positive=positive, alpha=alpha, min_stratum=min_stratum,
                min_effect=min_effect, min_evidence=min_evidence)


# ---- header -----------------------------------------------------------------------------
st.title("🔎 Paradox Hunter")
st.markdown("**An automated engine that finds where the overall trend in your data is lying to you** "
            "(Simpson's paradox and confounding reversals).")

# ---- sidebar: data + settings -----------------------------------------------------------
sb = st.sidebar
sb.header("1. Data")
source = sb.radio("Source", ["Sample dataset", "Upload CSV"], horizontal=True)
default_tier = "suggestive"
if source == "Sample dataset":
    name = sb.selectbox("Dataset", list(datasets.SAMPLES))
    fn, default_target = datasets.SAMPLES[name]
    df = fn()
    default_tier = SAMPLE_DEFAULTS[name]
else:
    up = sb.file_uploader("CSV file", type="csv")
    if up is None:
        st.info("Upload a CSV in the sidebar, or switch to a sample dataset. "
                "You need one binary outcome column (e.g. churned, admitted, survived).")
        st.stop()
    df, name, default_target = pd.read_csv(up), up.name, None

sb.header("2. Outcome")
cols = list(df.columns)
target = sb.selectbox("Target column", cols, index=cols.index(default_target) if default_target in cols else len(cols) - 1)
uniq = df[target].dropna().unique()
positive = None
if len(uniq) == 2:
    opts = sorted(uniq, key=str)
    positive = sb.selectbox("Which value counts as the 'outcome'?", opts, index=len(opts) - 1)
elif pd.api.types.is_numeric_dtype(df[target]):
    sb.caption("Numeric target with >2 values: it will be split at the median.")
else:
    sb.error("Target must be binary (2 distinct values) or numeric.")
    st.stop()

sb.header("3. Strictness")
min_evidence = sb.select_slider("Show findings at least this strong",
                                ["exploratory", "suggestive", "confirmed"], value=default_tier)
with sb.expander("Advanced"):
    alpha = st.slider("FDR level (Benjamini-Hochberg)", 0.01, 0.20, 0.05, 0.01)
    min_stratum = st.number_input("Min rows per subgroup", 5, 500, 20)
    min_effect = st.slider("Min size of overall effect (|log OR|)", 0.0, 0.5, 0.10, 0.01)

run = sb.button("Hunt for paradoxes", type="primary")
key = (name, target, str(positive), alpha, int(min_stratum), min_effect, min_evidence)
if source == "Sample dataset" or run:
    st.session_state["result"] = (key, run_scan(df, target, positive, alpha, int(min_stratum),
                                                min_effect, min_evidence))
if "result" not in st.session_state:
    st.info("Pick your outcome column, then press **Hunt for paradoxes**.")
    st.stop()
rkey, (findings, summary) = st.session_state["result"]
if rkey != key:
    st.warning("Settings changed since the last scan - press **Hunt for paradoxes** to refresh.")

# ---- summary ----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows scanned", f"{summary['rows']:,}")
m2.metric("Columns", summary["columns"])
m3.metric("Feature pairs tested", summary["tests_run"])
m4.metric("Paradoxes found", summary["paradoxes_found"])

with st.expander("Preview data"):
    show_df(df.head(50))

if not findings:
    st.success("No reversals found at this strictness. Your aggregate trends look trustworthy - "
               "or try lowering the strictness in the sidebar to see weaker, exploratory patterns.")
    st.stop()

# ---- findings table + detail ------------------------------------------------------------
st.subheader("Ranked findings")
show_df(findings_to_frame(findings))

idx = st.selectbox("Inspect a finding", range(len(findings)),
                   format_func=lambda i: f"#{i + 1}  {findings[i].exposure}  ×  {findings[i].stratifier}   "
                                         f"(score {findings[i].score:.0f})")
f = findings[idx]

st.subheader(f"{f.exposure}  ×  {f.stratifier}")
st.markdown(f"Evidence: {EVIDENCE_BADGE[f.evidence]} &nbsp;|&nbsp; "
            f"Reversal: **{'full' if f.full_reversal else 'partial'}** &nbsp;|&nbsp; "
            f"Paradox score: **{f.score:.0f}/100**")

c1, c2, c3 = st.columns(3)
c1.metric("Overall odds ratio", f"{f.crude_or:.2f}", f"{f.crude_direction}s outcome", delta_color="off")
c2.metric(f"Adjusted for {f.stratifier}", f"{f.adj_or:.2f}", f"{f.adj_direction} outcome", delta_color="off")
c3.metric("Subgroups agreeing with adjusted", f"{f.consistency:.0%}")

left, right = st.columns(2)
ov = overall_rates(f)
with left:
    st.markdown("##### ❌ What the aggregate chart says")
    show_chart(alt.Chart(ov).mark_bar().encode(
        x=alt.X("group:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title=f"P({target})", axis=alt.Axis(format="%"), scale=alt.Scale(domainMin=0)),
        color=alt.Color("group:N", legend=None),
        tooltip=["group", alt.Tooltip("rate:Q", format=".1%"), "n"]).properties(height=320))
with right:
    st.markdown(f"##### ✅ What happens inside each {f.stratifier} group")
    show_chart(alt.Chart(strata_long(f)).mark_bar().encode(
        x=alt.X("stratum:N", title=f.stratifier, axis=alt.Axis(labelAngle=0)),
        xOffset="group:N",
        y=alt.Y("rate:Q", title=f"P({target})", axis=alt.Axis(format="%"), scale=alt.Scale(domainMin=0)),
        color=alt.Color("group:N", title=None, legend=alt.Legend(orient="bottom")),
        tooltip=["stratum", "group", alt.Tooltip("rate:Q", format=".1%"), "n"]).properties(height=320))

st.markdown("#### Plain-English verdict")
st.info(f.explanation)

with st.expander("Subgroup table"):
    show_df(strata_table(f))

# ---- downloads --------------------------------------------------------------------------
d1, d2 = st.columns(2)
d1.download_button("⬇ Download report (.md)", make_report(findings, summary, name),
                   file_name="paradox_report.md", mime="text/markdown")
d2.download_button("⬇ Download findings (.csv)", findings_to_frame(findings).to_csv(index=False),
                   file_name="paradox_findings.csv", mime="text/csv")

with st.expander("How does this work?"):
    st.markdown("""
**Simpson's paradox** = a trend that appears in aggregated data reverses once you split by a hidden variable.

For every pair *(exposure X, stratifier Z)* in your data the engine:
1. Computes the **overall** odds ratio of X on the outcome (chi-square test).
2. Computes the **adjusted** odds ratio across the groups of Z using the **Mantel-Haenszel** estimator and the **Cochran-Mantel-Haenszel** test.
3. Flags the pair if the **direction flips**, and measures how many subgroups agree with the flip.
4. Corrects **all** tests with **Benjamini-Hochberg FDR**, so scanning hundreds of pairs doesn't invent paradoxes.

**Evidence tiers:** *Confirmed* = both effects significant. *Suggestive* = the overall trend is real but the adjusted effect is not significant (the trend is explained away). *Exploratory* = neither clears the bar (e.g. small samples).

⚠️ Reversals show **association structure, not proof of causation**. Deciding whether to trust the aggregate or the subgroups needs domain knowledge about which variable is the true confounder.
""")
