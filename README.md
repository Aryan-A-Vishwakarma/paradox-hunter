# 🔎 Paradox Hunter

**An automated engine that scans any tabular dataset and finds where the overall trend is lying to you.**

[Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox) is when a trend in aggregated data *reverses* once you split it by a hidden variable. Analysts check for it by hand, if at all. Paradox Hunter checks **every feature pair automatically**, ranks the reversals, controls false discoveries, and writes a plain-English *"don't trust this chart"* report.

## Results on real data

| Dataset | Headline (overall) | After adjusting | Verdict |
|---|---|---|---|
| **UC Berkeley admissions, 1973** (n=4,526) | Men admitted far more often (OR **1.84**, p≈1e-21) | OR **0.90** (not significant) once you account for *department* | Trend explained away by department choice |
| **Kidney stone treatment, 1986** (n=700) | Treatment B looks better (OR **1.34**) | OR **0.69**, B worse in **100%** of subgroups | Full reversal; small sample, so flagged *exploratory* |
| **Synthetic hospital** (planted + 4 decoy columns) | Hospital B looks deadlier (OR **1.68**) | OR **0.61** within each severity level | Found at q≈1e-9; **all decoys correctly ignored** |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py                                   # interactive dashboard
python -m paradox_hunter --sample berkeley             # CLI on a built-in dataset
python -m paradox_hunter mydata.csv --target churned --out report.md
python -m unittest discover -s tests                   # run the 11 tests
```

## How it works

For every pair *(exposure X, stratifier Z)* in the data:

1. **Overall effect**: 2×2 odds ratio of X on the binary outcome + chi-square test.
2. **Adjusted effect**: Mantel-Haenszel common odds ratio across the strata of Z (Robins-Breslow-Greenland variance) + Cochran-Mantel-Haenszel test. *Implemented from scratch with NumPy/SciPy and unit-tested against the published Berkeley value (0.90).*
3. **Reversal check**: does the direction flip? What share of the data's subgroups agree with the flip? (*full* vs *partial* reversal)
4. **Multiple-testing control**: Benjamini-Hochberg FDR over **all** pairs tested, not just the interesting ones, so scanning hundreds of pairs doesn't manufacture paradoxes.
5. **Evidence tiers**: *Confirmed* (both effects significant), *Suggestive* (overall trend real, adjusted effect not significant, i.e. explained away), *Exploratory* (neither significant, e.g. small n).
6. **Paradox score (0-100)**: reversal strength × subgroup consistency × evidence weight.
7. **Explanation**: a generated narrative naming the confounder and *why* it distorts the aggregate (which subgroup is over-represented in the exposed group, and the spread in baseline rates).

Automatic preprocessing: ID-like columns dropped, continuous features median-split as exposures and quartile-binned as stratifiers, multi-level categoricals handled one-vs-rest, missing values handled per pair.

## Validation

* **False-positive test**: 15 random-noise datasets → **0** reported paradoxes.
* **Negative control**: a genuine, consistent effect with no confounding is *not* flagged.
* **Planted-signal test**: synthetic data with a known paradox + decoys → exactly the planted one is returned.
* **Scale**: 100k rows × 15 columns (644 tests) in ~10 s.

## Project structure

```
app.py                      Streamlit dashboard
paradox_hunter/engine.py    detection engine (stats, FDR, scoring, explanations)
paradox_hunter/viz.py       chart-data helpers
paradox_hunter/datasets.py  Berkeley, kidney stones, synthetic hospital
paradox_hunter/__main__.py  CLI
tests/test_engine.py        unit + statistical validation tests
data/                       sample CSVs
```

## Limitations (read these before trusting a result)

* Binary outcomes only (numeric targets are median-split).
* Reversals reveal *association structure*, **not causation**. Whether to trust the aggregate or the subgroups depends on which variable is the true confounder, which needs domain knowledge. (In the kidney-stone case, stone size is a confounder; in other cases the "stratifier" could be a *mediator*, and adjusting for it would be wrong.)
* Only single-variable stratification is scanned (no Z1×Z2 combinations yet).

## Ideas to extend

Multi-variable stratification · causal-graph (DAG) input to decide which side to trust · continuous outcomes (regression-coefficient sign flips) · LLM-written narratives · permutation-based p-values.
