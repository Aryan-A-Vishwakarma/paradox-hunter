"""Chart-data helpers (pure pandas, so they can be tested without any UI framework)."""
import numpy as np
import pandas as pd

from .engine import Finding


def group_labels(f: Finding) -> tuple[str, str]:
    return f.exposure, f"NOT ({f.exposure})"


def overall_rates(f: Finding) -> pd.DataFrame:
    """Aggregate outcome rate for exposed vs unexposed - the 'misleading' headline chart."""
    s = f.strata
    lab_e, lab_u = group_labels(f)
    succ_e = (s.exposed_rate.fillna(0) * s.n_exposed).sum()
    succ_u = (s.unexposed_rate.fillna(0) * s.n_unexposed).sum()
    return pd.DataFrame({
        "group": [lab_e, lab_u],
        "rate": [succ_e / max(s.n_exposed.sum(), 1), succ_u / max(s.n_unexposed.sum(), 1)],
        "n": [int(s.n_exposed.sum()), int(s.n_unexposed.sum())],
    })


def strata_long(f: Finding) -> pd.DataFrame:
    """Per-stratum outcome rates in long format - the 'truth' chart."""
    s = f.strata
    lab_e, lab_u = group_labels(f)
    a = pd.DataFrame({"stratum": s.stratum, "group": lab_e, "rate": s.exposed_rate, "n": s.n_exposed})
    b = pd.DataFrame({"stratum": s.stratum, "group": lab_u, "rate": s.unexposed_rate, "n": s.n_unexposed})
    out = pd.concat([a, b], ignore_index=True)
    return out[out.n > 0].reset_index(drop=True)


def strata_table(f: Finding) -> pd.DataFrame:
    """Human-readable per-stratum table with a flag showing which strata flip the headline."""
    s = f.strata.copy()
    crude_sign = np.sign(np.log(f.crude_or))
    s["opposes_headline"] = np.sign(s.log_or) == -crude_sign
    out = s[["stratum", "n", "n_exposed", "exposed_rate", "unexposed_rate", "risk_diff", "opposes_headline"]]
    return out.rename(columns={"exposed_rate": "rate (exposed)", "unexposed_rate": "rate (unexposed)",
                               "risk_diff": "difference"})
