# 🔎 Paradox Hunter

### Sometimes the average lies. This tool catches it.

Paradox Hunter reads a spreadsheet (CSV), checks every column, and warns you when the **overall result is misleading**, then explains why in plain English.

<!-- After you take a screenshot of the dashboard, save it as screenshot.png in this folder,
     then delete this comment line and the closing arrow line below:
![Paradox Hunter dashboard](screenshot.png)
-->

---

## 🧠 The idea in 60 seconds

Two doctors, **Amit** and **Riya**, each treat 100 patients. Some cases are *easy* and some are *hard*.

| | Easy cases | Hard cases | **Overall** |
|---|---|---|---|
| **Amit** | 9 of 10 saved (**90%**) | 45 of 90 saved (**50%**) | **54%** |
| **Riya** | 72 of 90 saved (**80%**) | 4 of 10 saved (**40%**) | **76%** |

- Inside **easy** cases, Amit is better (90% vs 80%).
- Inside **hard** cases, Amit is better (50% vs 40%).
- But **overall**, Riya looks much better (76% vs 54%).

How? Amit was given mostly *hard* patients, and Riya mostly *easy* ones. The overall average mixes two groups that aren't comparable, so it points to the wrong doctor.

This is called **Simpson's paradox**. It has caused wrong conclusions in real medical studies, hiring audits, and business reports.

**Paradox Hunter finds these situations automatically**, so you don't have to guess which column to check.

---

## ✅ What it does

You give it a CSV and name the outcome you care about (survived, admitted, churned...). It then:

1. 🔍 **Tries every column** as a possible hidden factor
2. ⚖️ **Compares** the overall result with the result inside each group
3. 🚩 **Raises a flag** when the two point in opposite directions
4. 🎲 **Rules out luck** with statistical tests, so random noise doesn't create fake alarms
5. 📝 **Explains** the problem in plain English and ranks findings by strength

---

## 📊 Results on real data

| Dataset | What the overall numbers say | What's really going on |
|---|---|---|
| **UC Berkeley admissions (1973)** | Men were admitted far more often than women | The gap shrinks to almost nothing once you compare **within each department**. Women applied more to harder departments. |
| **Kidney stone treatment (1986)** | Treatment B looks better (83% vs 78%) | Treatment A is better for **both** small and large stones. A was used on harder, larger stones. |
| **Synthetic hospital (planted test)** | Hospital B looks deadlier | B is actually safer at **every** severity level. It just receives sicker patients. The tool found this and correctly **ignored 4 decoy columns**. |

<details>
<summary>The exact numbers (odds ratios)</summary>

| Dataset | Overall | After adjusting | Evidence level |
|---|---|---|---|
| Berkeley (n=4,526) | 1.84 | 0.90 | Suggestive |
| Kidney stones (n=700) | 1.34 | 0.69 | Exploratory (sample is small) |
| Synthetic hospital (n=6,000) | 1.68 | 0.61 | Confirmed |

An odds ratio above 1 means "more likely", below 1 means "less likely". When the number crosses from above 1 to below 1, the effect has flipped.
</details>

---

## 🚀 Try it yourself

**1. Install**

```bash
git clone https://github.com/Aryan-A-Vishwakarma/paradox-hunter.git
cd paradox-hunter
pip install -r requirements.txt
```

**2. Open the dashboard** (easiest)

```bash
streamlit run app.py
```

Pick a sample dataset in the sidebar, or upload your own CSV.

**3. Or use the command line**

```bash
# built-in example
python -m paradox_hunter --sample berkeley

# your own file (the target is the yes/no column you care about)
python -m paradox_hunter mydata.csv --target churned --out report.md
```

**4. Run the tests**

```bash
python -m unittest discover -s tests
```

You should see `Ran 11 tests ... OK`.

---

## 📖 How to read the dashboard

| You see | It means |
|---|---|
| **Left chart: "What the aggregate says"** | The overall picture, the one that can mislead you |
| **Right chart: "What happens inside each group"** | The fairer comparison |
| **Odds ratio (overall vs adjusted)** | Above 1 = raises the outcome, below 1 = lowers it. A flip across 1 is the paradox. |
| **🟢 Confirmed** | Both the overall trend and the reversal are statistically solid |
| **🟠 Suggestive** | The overall trend is real, but it disappears once you account for the hidden factor |
| **⚪ Exploratory** | Looks like a flip, but there isn't enough data to be sure |
| **Paradox score (0-100)** | How strong and how consistent the reversal is |

---

## 🔬 How it works (technical details)

For every pair of columns *(exposure X, grouping Z)*:

1. **Overall effect:** 2×2 odds ratio of X on the outcome, with a chi-square test.
2. **Adjusted effect:** Mantel-Haenszel common odds ratio across the groups of Z, with the Cochran-Mantel-Haenszel significance test. *Written from scratch with NumPy/SciPy and checked against the published Berkeley value of 0.90.*
3. **Reversal check:** does the direction flip, and what share of the data's subgroups agree with the flip? (*full* vs *partial* reversal)
4. **Multiple-testing control:** Benjamini-Hochberg false discovery rate over **all** pairs tested, not only the interesting ones.
5. **Evidence tiers** (Confirmed / Suggestive / Exploratory) and a **0-100 score** based on reversal size, subgroup agreement, and evidence strength.
6. **Plain-English explanation** naming the hidden factor and why it distorts the average.

Automatic cleaning: ID-like columns are dropped, numeric columns are split at the median (as a cause) or into quartiles (as a grouping), multi-value categories are handled one-vs-rest, and missing values are handled per pair.

---

## 🧪 How it was tested

| Test | Result |
|---|---|
| 15 datasets of pure random noise | **0** false alarms |
| A real effect with no hidden factor | Correctly **not** flagged |
| Synthetic data with one planted paradox + 4 decoys | Found exactly the planted one |
| Adjusted odds ratio for Berkeley | Matches the published value (0.90) |
| 100,000 rows × 15 columns (644 tests) | About 10 seconds |

---

## 📁 Project structure

```
app.py                       Streamlit dashboard
paradox_hunter/
  engine.py                  detection engine (statistics, scoring, explanations)
  viz.py                     helpers that prepare chart data
  datasets.py                Berkeley, kidney stones, synthetic hospital
  __main__.py                command-line tool
tests/test_engine.py         11 automated tests
data/                        sample CSV files
```

---

## ⚠️ Limitations

- **Association, not causation.** The tool shows *that* two views disagree and *which column* causes it. It cannot tell you which view is correct. That needs knowledge of what the data means. (Sometimes the "hidden factor" is actually a *result* of the cause, and splitting by it would be the wrong move.)
- Works with **yes/no outcomes** (numeric outcomes are split at the median).
- Checks **one grouping column at a time**, not combinations.

---

## 🔭 Possible next steps

- Check combinations of two grouping columns
- Let the user draw a causal diagram to decide which view to trust
- Support numeric outcomes (such as price or score)
- Add permutation-based significance tests

---

## 👤 Author

**Aryan Vishwakarma**, AI & ML engineering student
GitHub: [@Aryan-A-Vishwakarma](https://github.com/Aryan-A-Vishwakarma)

If you found this useful, a ⭐ on the repo helps a lot.
