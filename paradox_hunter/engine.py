"""
Paradox Hunter - core detection engine.

Finds Simpson's-paradox-style reversals in tabular data with a binary outcome:
cases where an exposure X looks like it pushes the outcome Y one way overall,
but after accounting for a stratifier Z the effect points the other way.

Statistics used
---------------
* Crude association:     2x2 odds ratio of (X, Y) + chi-square test (Yates).
* Adjusted association:  Mantel-Haenszel common odds ratio across strata of Z,
                         with the Robins-Breslow-Greenland variance and the
                         Cochran-Mantel-Haenszel (CMH) test.
* Multiple testing:      Benjamini-Hochberg FDR across ALL (X, Z) pairs tested,
                         not just the ones that happened to look interesting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def binarize_target(series: pd.Series, positive=None) -> pd.Series:
    """Return a 0/1 integer Series. Numeric targets with >2 values are median-split."""
    s = series.dropna()
    uniq = s.unique()
    if positive is not None:
        out = (series == positive).astype(float)
    elif len(uniq) == 2:
        # prefer 1 / True / 'yes' style positives, else the second sorted value
        try:
            positive = sorted(uniq)[-1]
        except TypeError:
            positive = uniq[0]
        out = (series == positive).astype(float)
    elif pd.api.types.is_numeric_dtype(series):
        out = (series > series.median()).astype(float)
    else:
        raise ValueError("Target must be binary (or numeric, which gets median-split). "
                         "Pass `positive=` to pick the class to treat as the outcome.")
    out[series.isna()] = np.nan
    return out


def _is_idlike(s: pd.Series) -> bool:
    return (not pd.api.types.is_float_dtype(s)) and s.nunique() > 0.5 * len(s) and len(s) > 20


def _make_stratifier(s: pd.Series, max_levels: int, n_bins: int = 4) -> pd.Series | None:
    """Turn a column into a categorical stratifier, or None if unusable."""
    s = s.copy()
    nun = s.nunique(dropna=True)
    if nun < 2:
        return None
    if pd.api.types.is_numeric_dtype(s) and nun > max_levels:
        try:
            return pd.qcut(s, q=n_bins, duplicates="drop").astype(str).where(s.notna())
        except ValueError:
            return None
    if nun > max_levels or _is_idlike(s):
        return None
    return s.astype(str).where(s.notna())


def _make_exposures(s: pd.Series, name: str, max_levels: int, min_share: float = 0.05):
    """Yield (label, 0/1 Series) exposure candidates from a column."""
    nun = s.nunique(dropna=True)
    if nun < 2 or _is_idlike(s):
        return
    if pd.api.types.is_numeric_dtype(s) and nun > 2:
        if nun > max_levels:                         # continuous: median split
            med = s.median()
            b = (s > med).astype(float).where(s.notna())
            if 0.05 < b.mean() < 0.95:
                yield f"{name} > {med:g}", b
            return
    if nun == 2:
        levels = sorted(s.dropna().unique(), key=str)
        top = levels[-1]
        yield f"{name} = {top}", (s == top).astype(float).where(s.notna())
        return
    if nun <= max_levels:                            # multi-level: one-vs-rest
        shares = s.value_counts(normalize=True)
        for lvl in shares[shares >= min_share].index[:5]:
            yield f"{name} = {lvl}", (s == lvl).astype(float).where(s.notna())


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _table(x: np.ndarray, y: np.ndarray) -> tuple[int, int, int, int]:
    a = int(((x == 1) & (y == 1)).sum())   # exposed, outcome
    b = int(((x == 1) & (y == 0)).sum())   # exposed, no outcome
    c = int(((x == 0) & (y == 1)).sum())   # unexposed, outcome
    d = int(((x == 0) & (y == 0)).sum())   # unexposed, no outcome
    return a, b, c, d


def _log_or(a, b, c, d) -> float:
    """Log odds ratio with Haldane-Anscombe 0.5 correction when any cell is zero."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    return float(np.log((a * d) / (b * c)))


def _rd(a, b, c, d) -> float:
    n1, n0 = a + b, c + d
    if n1 == 0 or n0 == 0:
        return float("nan")
    return a / n1 - c / n0


def _mantel_haenszel(tables: list[tuple[int, int, int, int]]):
    """Return (OR_MH, CMH p-value) over a list of 2x2 tables."""
    num = den = 0.0
    sum_a = sum_e = sum_v = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        n1, n0, m1, m0 = a + b, c + d, a + c, b + d
        sum_a += a
        sum_e += n1 * m1 / n
        sum_v += n1 * n0 * m1 * m0 / (n ** 2 * (n - 1))
    if num == 0 and den == 0:
        return float("nan"), 1.0
    if den == 0:
        or_mh = float("inf")
    elif num == 0:
        or_mh = 0.0
    else:
        or_mh = num / den
    if sum_v <= 0:
        return or_mh, 1.0
    chi2 = (abs(sum_a - sum_e) - 0.5) ** 2 / sum_v if abs(sum_a - sum_e) > 0.5 else 0.0
    return or_mh, float(stats.chi2.sf(chi2, 1))


def _crude_p(a, b, c, d) -> float:
    tbl = np.array([[a, b], [c, d]])
    if tbl.sum(axis=0).min() == 0 or tbl.sum(axis=1).min() == 0:
        return 1.0
    return float(stats.chi2_contingency(tbl, correction=True)[1])


def _cramers_v(u: pd.Series, v: pd.Series) -> float:
    ct = pd.crosstab(u, v)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if n else 0.0


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(ranked, 1.0)
    return q


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    exposure: str
    stratifier: str
    outcome: str
    n: int
    crude_or: float
    adj_or: float
    crude_rd: float
    p_crude: float
    p_adj: float
    consistency: float            # share of (weighted) strata agreeing with the adjusted direction
    full_reversal: bool           # every informative stratum disagrees with the crude direction
    strata: pd.DataFrame = field(repr=False)
    confounding: dict = field(repr=False, default_factory=dict)
    q_crude: float = float("nan")
    q_adj: float = float("nan")
    evidence: str = "exploratory"   # confirmed | suggestive | exploratory
    score: float = 0.0
    explanation: str = ""

    @property
    def q(self) -> float:
        return max(self.q_crude, self.q_adj)

    @property
    def crude_direction(self) -> str:      # bare verb: "appears to increase"
        return "increase" if self.crude_or > 1 else "decrease"

    @property
    def adj_direction(self) -> str:        # 3rd person: "it actually increases"
        return "increases" if self.adj_or > 1 else "decreases"

    def to_row(self) -> dict:
        return {
            "score": round(self.score, 1),
            "exposure": self.exposure,
            "stratified_by": self.stratifier,
            "overall_OR": round(self.crude_or, 2),
            "adjusted_OR": round(self.adj_or, 2),
            "evidence": self.evidence,
            "reversal": "full" if self.full_reversal else "partial",
            "strata_agreeing": f"{self.consistency:.0%}",
            "q_overall": float(f"{self.q_crude:.2e}"),
            "q_adjusted": float(f"{self.q_adj:.2e}"),
            "n": self.n,
        }


# --------------------------------------------------------------------------- #
# Main scan
# --------------------------------------------------------------------------- #


def _analyse_pair(xlabel, x, zname, z, y, outcome_name, min_stratum, min_effect=0.10) -> tuple | None:
    """Return (Finding-or-None, p_crude, p_adj) for one (exposure, stratifier) pair."""
    df = pd.DataFrame({"x": x, "z": z, "y": y}).dropna()
    if len(df) < 30:
        return None
    a, b, c, d = _table(df.x.values, df.y.values)
    if min(a + b, c + d, a + c, b + d) == 0:
        return None

    tables, rows = [], []
    for lvl, g in df.groupby("z", observed=True):
        ta = _table(g.x.values, g.y.values)
        n1, n0 = ta[0] + ta[1], ta[2] + ta[3]
        rows.append({
            "stratum": lvl, "n": len(g), "n_exposed": n1, "n_unexposed": n0,
            "exposed_rate": ta[0] / n1 if n1 else np.nan,
            "unexposed_rate": ta[2] / n0 if n0 else np.nan,
            "risk_diff": _rd(*ta),
            "log_or": _log_or(*ta) if (n1 > 0 and n0 > 0) else np.nan,
        })
        # informative strata only: both exposure arms present and some outcome variation
        if len(g) >= min_stratum and n1 > 0 and n0 > 0 and 0 < (ta[0] + ta[2]) < len(g):
            tables.append(ta)
    strata = pd.DataFrame(rows)
    if len(tables) < 2:
        return None

    crude_lor = _log_or(a, b, c, d)
    p_crude = _crude_p(a, b, c, d)
    or_mh, p_adj = _mantel_haenszel(tables)
    if not np.isfinite(or_mh) or or_mh <= 0:
        return None
    adj_lor = float(np.log(or_mh))

    # does the sign actually flip?
    # (and ignore flips where the overall effect is negligible: OR ~ 1.00 -> 0.97 is noise)
    if np.sign(crude_lor) == np.sign(adj_lor) or abs(crude_lor) < min_effect:
        return None, p_crude, p_adj

    # how many strata (weighted by size) agree with the adjusted direction?
    info = strata[strata.n >= min_stratum].dropna(subset=["log_or"])
    info = info[(info.n_exposed > 0) & (info.n_unexposed > 0)]
    agree = np.sign(info.log_or) == np.sign(adj_lor)
    consistency = float((info.n * agree).sum() / info.n.sum()) if len(info) else 0.0
    opposite_crude = np.sign(info.log_or) != np.sign(crude_lor)
    full = bool(len(info) > 0 and opposite_crude.all())

    # confounding diagnostics: how strongly Z is tied to X and to Y
    conf = {
        "z_vs_exposure": _cramers_v(df.z, df.x),
        "z_vs_outcome": _cramers_v(df.z, df.y),
    }
    f = Finding(
        exposure=xlabel, stratifier=zname, outcome=outcome_name, n=len(df),
        crude_or=float(np.exp(crude_lor)), adj_or=float(or_mh),
        crude_rd=_rd(a, b, c, d), p_crude=p_crude, p_adj=p_adj,
        consistency=consistency, full_reversal=full, strata=strata, confounding=conf,
    )
    return f, p_crude, p_adj


_EVIDENCE_WEIGHT = {"confirmed": 1.0, "suggestive": 0.6, "exploratory": 0.3}


def _score(f: Finding) -> float:
    """0-100. Rewards big reversals, consistent strata, and stronger evidence tiers."""
    strength = abs(np.log(f.crude_or)) + abs(np.log(f.adj_or))   # distance across the flip
    w = _EVIDENCE_WEIGHT[f.evidence]
    if f.evidence == "confirmed":                                # q=1e-3 -> full marks
        w *= max(0.5, min(1.0, -np.log10(max(f.q, 1e-300)) / 3.0))
    bonus = 1.0 if f.full_reversal else 0.85
    return float(100 * np.tanh(strength) * f.consistency * bonus * w)


def _explain(f: Finding) -> str:
    s = f.strata.dropna(subset=["risk_diff"])
    crude_pct = f.crude_rd * 100
    verb_c, verb_a = f.crude_direction, f.adj_direction
    out = f.outcome
    txt = (
        f"Overall, **{f.exposure}** appears to **{verb_c}** the chance of *{out}* "
        f"(odds ratio {f.crude_or:.2f}, a {abs(crude_pct):.1f}-point difference in rate). "
    )
    if f.full_reversal:
        txt += (f"But within **every** group of *{f.stratifier}*, the effect points the other way. ")
    else:
        txt += (f"But within most groups of *{f.stratifier}* ({f.consistency:.0%} of the data), "
                f"the effect points the other way. ")
    txt += (f"After adjusting for *{f.stratifier}* (Mantel-Haenszel), it actually **{verb_a}** "
            f"*{out}* (adjusted OR {f.adj_or:.2f}). ")
    txt += {
        "confirmed": "Both the overall trend and the reversed adjusted trend are statistically significant. ",
        "suggestive": ("The overall trend is statistically real, but the adjusted effect is too small to be "
                       "significant - so the headline trend is best explained away by the confounder. "),
        "exploratory": ("Neither effect clears the significance bar (small sample), so treat this as a "
                        "pattern worth checking, not a proven result. "),
    }[f.evidence]

    # name the culprit: the stratum where exposed and unexposed groups differ most in share
    if len(s) >= 2 and s.n_exposed.sum() > 0 and s.n_unexposed.sum() > 0:
        s = s.assign(exp_share=s.n_exposed / s.n_exposed.sum(),
                     unexp_share=s.n_unexposed / s.n_unexposed.sum(),
                     base=(s.exposed_rate.fillna(0) * s.n_exposed + s.unexposed_rate.fillna(0) * s.n_unexposed)
                          / (s.n_exposed + s.n_unexposed))
        top = s.assign(gap=(s.exp_share - s.unexp_share).abs()).sort_values("gap", ascending=False).iloc[0]
        txt += (f"**Why it happens:** the two groups are not spread evenly across *{f.stratifier}*. "
                f"For example '{top.stratum}' holds {top.exp_share:.0%} of the exposed rows but "
                f"{top.unexp_share:.0%} of the unexposed rows, and baseline *{out}* rates vary a lot "
                f"between groups ({s.base.min():.0%} to {s.base.max():.0%}). "
                f"*{f.stratifier}* is the hidden confounder - **don't trust the aggregate chart.**")
    return txt


_EVIDENCE_ORDER = {"exploratory": 0, "suggestive": 1, "confirmed": 2}


def scan(
    df: pd.DataFrame,
    target: str,
    positive=None,
    alpha: float = 0.05,
    min_stratum: int = 20,
    min_effect: float = 0.10,
    max_levels: int = 12,
    max_rows: int | None = 200_000,
    exclude: Iterable[str] = (),
    min_evidence: str = "suggestive",
    seed: int = 0,
) -> tuple[list[Finding], dict]:
    """
    Scan every (exposure, stratifier) pair for sign-reversal paradoxes.

    min_evidence: "confirmed" | "suggestive" | "exploratory" - lowest tier to report.
    min_effect:   minimum |log odds ratio| of the overall trend for a flip to count (0.10 ~ OR 1.1).

    Returns (findings sorted by score, summary dict with test counts).
    """
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not in dataframe")
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    y = binarize_target(df[target], positive)
    cols = [c for c in df.columns if c != target and c not in set(exclude)]

    exposures = {c: list(_make_exposures(df[c], c, max_levels)) for c in cols}
    stratifiers = {c: _make_stratifier(df[c], max_levels) for c in cols}

    results = []   # (finding_or_None, p_crude, p_adj)
    for ecol, zcol in product(cols, cols):
        if ecol == zcol or stratifiers[zcol] is None:
            continue
        for xlabel, x in exposures[ecol]:
            r = _analyse_pair(xlabel, x, zcol, stratifiers[zcol], y, target, min_stratum, min_effect)
            if r is not None:
                results.append(r)

    n_tests = len(results)
    # FDR is computed over EVERY pair tested (not only the ones that flipped sign),
    # so the correction honestly reflects how many places we looked.
    q_crude = benjamini_hochberg(np.array([r[1] for r in results]))
    q_adj = benjamini_hochberg(np.array([r[2] for r in results]))

    findings: list[Finding] = []
    for (f, _, _), qc, qa in zip(results, q_crude, q_adj):
        if f is None:
            continue
        f.q_crude, f.q_adj = float(qc), float(qa)
        if qc <= alpha and qa <= alpha:
            f.evidence = "confirmed"
        elif qc <= alpha:
            f.evidence = "suggestive"
        else:
            f.evidence = "exploratory"
        if _EVIDENCE_ORDER[f.evidence] < _EVIDENCE_ORDER[min_evidence]:
            continue
        f.score = _score(f)
        f.explanation = _explain(f)
        findings.append(f)

    findings = _dedupe(findings)
    findings.sort(key=lambda f: f.score, reverse=True)
    summary = {
        "rows": len(df), "columns": len(cols), "tests_run": n_tests,
        "paradoxes_found": len(findings), "alpha": alpha, "min_evidence": min_evidence,
        "base_rate": float(np.nanmean(y)),
        "by_evidence": {k: sum(f.evidence == k for f in findings) for k in _EVIDENCE_ORDER},
    }
    return findings, summary


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Keep the best-scoring finding per (exposure column, stratifier) pair."""
    best: dict[tuple, Finding] = {}
    for f in findings:
        key = (f.exposure.split(" ")[0], f.stratifier)
        if key not in best or f.score > best[key].score:
            best[key] = f
    return list(best.values())


def findings_to_frame(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([f.to_row() for f in findings])


def make_report(findings: list[Finding], summary: dict, dataset_name: str = "dataset") -> str:
    """Markdown 'Don't trust this chart' report."""
    lines = [
        f"# Paradox Hunter report - {dataset_name}", "",
        f"Scanned **{summary['rows']:,} rows**, **{summary['columns']} columns**, "
        f"**{summary['tests_run']} exposure/stratifier tests** "
        f"(Benjamini-Hochberg FDR = {summary['alpha']}).",
        f"Found **{summary['paradoxes_found']}** reversal(s) at evidence level "
        f"'{summary.get('min_evidence', 'suggestive')}' or stronger.", "",
    ]
    for i, f in enumerate(findings, 1):
        lines += [
            f"## {i}. {f.exposure} x {f.stratifier}  (score {f.score:.0f}/100, {f.evidence})", "",
            f.explanation, "",
            f"| | Overall | Adjusted for {f.stratifier} |", "|---|---|---|",
            f"| Odds ratio | {f.crude_or:.2f} | {f.adj_or:.2f} |",
            f"| p-value | {f.p_crude:.2e} | {f.p_adj:.2e} |", "",
            f"Reversal type: {'full' if f.full_reversal else 'partial'} | "
            f"q(overall) = {f.q_crude:.2e} | q(adjusted) = {f.q_adj:.2e} | n = {f.n:,}", "",
        ]
    return "\n".join(lines)
