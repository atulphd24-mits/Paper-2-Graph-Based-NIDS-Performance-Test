"""
Best E-GraphSAGE endpoint for the 30-seed headline runs (PROTOCOL_FREEZE.md, section 6), chosen
per source from VALIDATION rows only, among the real endpoints (IP, IP:port; the random control
is excluded): highest mean design-weighted validation PR-AUC of the
inductive model at the main protocol over seeds 0-9. Test rows are never read.

Usage: python scripts/select_endpoint.py [--tag _final_egs_ind] [--seeds 0-9]
Writes outputs/paper2_final/reports/endpoint_selection.{json,csv}.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper2_final"
MAIN = "episodes_gap_1_min_embargo_5_min"
CONTROLS = {"random"}  # random endpoints are the control, never a candidate
COLUMN = re.compile(r"^E-GraphSAGE \((?P<endpoint>[^)]+)\)\|inductive$")


def validation_pr_auc(frame):
    """{endpoint: PR-AUC} on validation rows of one prediction file."""
    val = frame[frame["split"].astype(str) == "validation"]
    out = {}
    for col in frame.columns:
        m = COLUMN.match(col)
        if m:
            out[m["endpoint"]] = average_precision_score(val["label"], val[col], sample_weight=val["design_weight"])
    return out


def select(table):
    """table: rows (source, seed, endpoint, val_pr_auc) -> {source: best endpoint}."""
    table = table[~table["endpoint"].isin(CONTROLS)]
    mean = table.groupby(["source", "endpoint"])["val_pr_auc"].mean().reset_index()
    best = mean.sort_values(["source", "val_pr_auc", "endpoint"], ascending=[True, False, True])
    return best.groupby("source").first()["endpoint"].to_dict(), mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="_final_egs_ind")
    parser.add_argument("--seeds", default="0-9")
    args = parser.parse_args()
    lo, hi = map(int, args.seeds.split("-"))
    rows = []
    for source_dir in sorted((OUT / "predictions" / f"egraphsage{args.tag}" / "bigflow").iterdir()):
        for seed in range(lo, hi + 1):
            path = source_dir / f"seed{seed}_{MAIN}.parquet"
            frame = pd.read_parquet(path, columns=None)
            for endpoint, value in validation_pr_auc(frame).items():
                rows.append({"source": source_dir.name, "seed": seed, "endpoint": endpoint, "val_pr_auc": value})
    table = pd.DataFrame(rows)
    best, mean = select(table)
    mean.to_csv(OUT / "reports" / "endpoint_selection.csv", index=False)
    (OUT / "reports" / "endpoint_selection.json").write_text(json.dumps(
        {"rule": "max mean design-weighted validation PR-AUC, inductive, main protocol, IP or IP:port, seeds "
                 + args.seeds,
         "best_endpoint": best, "n_files": int(table.groupby(["source", "seed"]).ngroups)}, indent=2))
    print(json.dumps(best))


if __name__ == "__main__":
    main()
