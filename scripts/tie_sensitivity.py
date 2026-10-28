"""
Post-hoc sensitivity analysis (DEVIATION from the frozen protocol, supplement only) for the cells
whose alert threshold is tied with the maximum score (stats/threshold_ties.csv).

Under the frozen rule an alert is `score > tau`, tau = 99th percentile of validation benign scores.
If >= 1% of validation benign flows share the maximum score, tau equals that maximum and no flow can
alert: the model cannot operate at 1% FPR. This script recomputes those cells with `score >= tau`
and reports the FPR that rule actually reaches on test, next to the frozen-rule values.

Usage: python scripts/tie_sensitivity.py
Writes outputs/paper2_final/stats/tie_sensitivity_ge.csv and reports/tie_sensitivity_ge.md.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from smoke_compare import md

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "outputs/paper2_final/predictions"
STATS = ROOT / "outputs/paper2_final/stats"


def tie_cells():
    t = pd.read_csv(STATS / "threshold_ties.csv")
    g = t.groupby(["part", "source", "protocol", "score"])["tie_at_max"].sum()
    return g[g > 0].reset_index()[["part", "source", "protocol", "score"]]


def evaluate(frame, col):
    """Frozen (>) and sensitivity (>=) TPR on subsets (i) and (iii), and test FPR reached."""
    s = frame[col].to_numpy(float)
    split, y, w = frame["split"].astype(str).to_numpy(), frame["label"].to_numpy(), frame["design_weight"].to_numpy(float)
    tau = np.quantile(s[(split == "validation") & (y == 0)], 0.99)
    test = split == "test"
    out = {"tied": bool(tau >= s.max())}
    for rule, alert in ((">", s > tau), (">=", s >= tau)):
        out[f"FPR_test {rule}"] = alert[test & (y == 0)].mean()
        for name, mask in (("(iii)", test & (y == 1)), ("(i)", test & (y == 1) & frame["in_cross"].to_numpy())):
            out[f"TPR{name} {rule}"] = np.average(alert[mask], weights=w[mask]) if mask.any() else np.nan
    return out


def main():
    rows = []
    for c in tie_cells().itertuples():
        for f in sorted((P / c.part).rglob(f"seed*_{c.protocol}.parquet")):
            if f.parent.name != c.source:
                continue
            seed = int(re.match(r"seed(\d+)_", f.name)[1])
            frame = pd.read_parquet(f, columns=["split", "label", "design_weight", "in_cross", c.score])
            rows.append({"part": c.part, "source": c.source, "protocol": c.protocol, "score": c.score, "seed": seed,
                         **evaluate(frame, c.score)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(STATS / "tie_sensitivity_ge.csv", index=False)
    cols = [c for c in per_seed.columns if c.startswith(("TPR", "FPR"))]
    summary = per_seed.groupby(["part", "source", "protocol", "score"]).agg(
        seeds=("seed", "nunique"), tied_seeds=("tied", "sum"), **{c: (c, "mean") for c in cols}).reset_index()
    text = ["# Tie sensitivity (post-hoc deviation, supplement only)\n", __doc__.strip() + "\n",
            "Means over seeds. `tied_seeds`: seeds in which the threshold equals the maximum score. "
            "Frozen-rule values (`>`) in the main text are shown as \"cannot operate at 1% FPR (tied scores)\" "
            "for these cells.\n", md(summary, 3)]
    (ROOT / "outputs/paper2_final/reports/tie_sensitivity_ge.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
