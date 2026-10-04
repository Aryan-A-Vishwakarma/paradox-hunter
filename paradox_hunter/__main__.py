"""CLI:  python -m paradox_hunter data.csv --target churned   |   python -m paradox_hunter --sample berkeley"""
import argparse
import sys

import pandas as pd

from . import datasets, make_report, scan, findings_to_frame


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="paradox_hunter", description="Find Simpson's-paradox reversals in a CSV.")
    p.add_argument("csv", nargs="?", help="path to a CSV file")
    p.add_argument("--target", help="binary outcome column")
    p.add_argument("--positive", help="value of the target to treat as the outcome (default: auto)")
    p.add_argument("--sample", choices=["berkeley", "kidney", "hospital"], help="run on a built-in dataset")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR level (default 0.05)")
    p.add_argument("--min-evidence", default="suggestive", choices=["confirmed", "suggestive", "exploratory"])
    p.add_argument("--min-stratum", type=int, default=20)
    p.add_argument("--out", help="write a Markdown report to this path")
    a = p.parse_args(argv)

    if a.sample:
        fn, target = {"berkeley": datasets.SAMPLES["UC Berkeley admissions (1973)"],
                      "kidney": datasets.SAMPLES["Kidney stone treatment (1986)"],
                      "hospital": datasets.SAMPLES["Synthetic hospital (planted paradox + noise)"]}[a.sample]
        df, name = fn(), a.sample
    elif a.csv and a.target:
        df, target, name = pd.read_csv(a.csv), a.target, a.csv
    else:
        p.error("give either --sample NAME, or a CSV path plus --target COLUMN")

    positive = a.positive
    if positive is not None and positive in df[target].astype(str).unique():
        positive = df[target][df[target].astype(str) == positive].iloc[0]   # keep original dtype

    findings, summary = scan(df, target, positive=positive, alpha=a.alpha,
                             min_stratum=a.min_stratum, min_evidence=a.min_evidence)
    print(f"Scanned {summary['rows']:,} rows x {summary['columns']} cols | "
          f"{summary['tests_run']} tests | {summary['paradoxes_found']} paradox(es) found\n")
    if findings:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(findings_to_frame(findings).to_string(index=False))
        print("\nTop finding:\n" + findings[0].explanation.replace("**", "").replace("*", ""))
    else:
        print("No reversals found at this evidence level. Try --min-evidence exploratory.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(make_report(findings, summary, name))
        print(f"\nReport written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
