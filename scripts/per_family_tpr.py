"""
Per-family TPR at 1% FPR for every run, seed, score column and test subset, design-weighted within
each family (the result CSVs hold unweighted per-family recall, which over-weights small episodes in
the episode-stratified sample). Same threshold and rule as the frozen analysis: tau = 99th percentile
of the file's validation benign scores, alert = score > tau. Tied seeds (tau = maximum score) are
flagged, not dropped.

Also writes each family's share of the subset's test-attack design weight, so tables can show how much
each family contributes to the pooled value.

Usage: python scripts/per_family_tpr.py  -> outputs/paper2_final/stats/per_family_tpr.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs/paper2_final/predictions"
META = {"row", "split", "t_ms", "t_end_ms", "episode", "episode_start_ms", "mode", "family", "label",
        "design_weight", "in_cross", "in_fallback"}


def per_family(frame, col):
    s = frame[col].to_numpy(float)
    split, y = frame["split"].astype(str).to_numpy(), frame["label"].to_numpy()
    tau = float(np.quantile(s[(split == "validation") & (y == 0)], 0.99))
    tied = tau >= s.max()
    test_att = (split == "test") & (y == 1)
    fam = frame["family"].astype(str).to_numpy()
    w = frame["design_weight"].to_numpy(float)
    alert = s > tau
    subsets = {"(iii) all": test_att, "(i) cross-episode": test_att & frame["in_cross"].to_numpy(),
               "(ii) fallback": test_att & frame["in_fallback"].to_numpy()}
    rows = []
    for name, mask in subsets.items():
        total = w[mask].sum()
        if total == 0:
            continue
        for f in np.unique(fam[mask]):
            m = mask & (fam == f)
            rows.append({"subset": name, "family": f, "tpr": float(np.average(alert[m], weights=w[m])),
                         "weight_share": float(w[m].sum() / total), "test_flows": int(m.sum()),
                         "test_episodes": int(frame.loc[m, "episode"].nunique()), "tied": bool(tied)})
    return rows


def main():
    out = []
    for f in sorted(PRED.rglob("*.parquet")):
        part = f.parts[-4]
        if "_final_" not in part:
            continue
        dataset, source = f.parts[-3], f.parts[-2]
        m = re.match(r"seed(\d+)_(.+)\.parquet", f.name)
        seed, rest = int(m[1]), m[2]
        variant = next((v for v in ("split_validation", "reused_validation") if rest.endswith(v)), None)
        protocol = rest[: -len(variant) - 1] if variant else rest
        frame = pd.read_parquet(f)
        for col in [c for c in frame.columns if c not in META]:
            for r in per_family(frame, col):
                out.append({"part": part, "dataset": dataset, "source": source, "protocol": protocol,
                            "validation": (variant or "-").replace("_", " "), "score": col, "seed": seed, **r})
    d = pd.DataFrame(out)
    d.to_csv(ROOT / "outputs/paper2_final/stats/per_family_tpr.csv", index=False)
    print(len(d), "rows")


if __name__ == "__main__":
    main()
