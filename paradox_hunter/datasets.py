"""Built-in sample datasets, rebuilt row-by-row from published aggregate counts."""
import numpy as np
import pandas as pd


def _expand(rows: list[tuple], cols: list[str]) -> pd.DataFrame:
    """rows: (*category values, n_total, n_positive, outcome_name) -> row-level frame."""
    recs = []
    for *cats, total, pos in rows:
        recs += [(*cats, 1)] * pos + [(*cats, 0)] * (total - pos)
    return pd.DataFrame(recs, columns=cols)


def berkeley_admissions() -> pd.DataFrame:
    """UC Berkeley 1973 graduate admissions (Bickel, Hammel & O'Connell, 1975)."""
    # (department, gender, applicants, admitted)
    raw = [
        ("A", "Male", 825, 512), ("A", "Female", 108, 89),
        ("B", "Male", 560, 353), ("B", "Female", 25, 17),
        ("C", "Male", 325, 120), ("C", "Female", 593, 202),
        ("D", "Male", 417, 138), ("D", "Female", 375, 131),
        ("E", "Male", 191, 53),  ("E", "Female", 393, 94),
        ("F", "Male", 373, 22),  ("F", "Female", 341, 24),
    ]
    df = _expand([(d, g, n, a) for d, g, n, a in raw], ["department", "gender", "admitted"])
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def kidney_stones() -> pd.DataFrame:
    """Charig et al. 1986: open surgery (A) vs percutaneous nephrolithotomy (B)."""
    raw = [
        ("A: open surgery", "small (<2cm)", 87, 81), ("A: open surgery", "large (>=2cm)", 263, 192),
        ("B: percutaneous", "small (<2cm)", 270, 234), ("B: percutaneous", "large (>=2cm)", 80, 55),
    ]
    df = _expand(raw, ["treatment", "stone_size", "success"])
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def synthetic_hospital(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """
    Synthetic hospital data with a planted paradox AND decoy columns.
    * Hospital B looks deadlier overall, but only because it receives sicker patients.
      Within every severity level, B actually has LOWER mortality.
    * age, insurance, and weekday are noise (should NOT be flagged).
    """
    rng = np.random.default_rng(seed)
    hospital = rng.choice(["A", "B"], n)
    p_severe = np.where(hospital == "B", 0.65, 0.20)
    severity = np.where(rng.random(n) < p_severe, "severe", "mild")
    base = np.where(severity == "severe", 0.30, 0.04)
    p_die = np.where(hospital == "B", base * 0.7, base)
    died = (rng.random(n) < p_die).astype(int)
    return pd.DataFrame({
        "hospital": hospital, "severity": severity,
        "age": rng.integers(18, 90, n),
        "insurance": rng.choice(["public", "private", "none"], n),
        "weekday": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
        "died": died,
    })


SAMPLES = {
    "UC Berkeley admissions (1973)": (berkeley_admissions, "admitted"),
    "Kidney stone treatment (1986)": (kidney_stones, "success"),
    "Synthetic hospital (planted paradox + noise)": (synthetic_hospital, "died"),
}
