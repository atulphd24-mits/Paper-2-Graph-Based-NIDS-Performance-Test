"""
Post-hoc (DEVIATION, decided 2026-10-20 after results were seen): how much of the random-row vs
episode-split gap (claim C4) is family mix, how much is threshold transfer, and how much is left.

From the saved predictions of the split comparison (HistGB, RF, MLP on flow/behaviour features,
seeds 0-9, BigFlow ToN-IoT and CSE-CIC-IDS2018), subset (iii) = all test attacks:

  family mix   (a) pooled: design weights, as in the frozen analysis;
               (b) reweighted: test-attack weights rescaled per family so every split has the
                   full-data family mix (total attack weight unchanged, benign untouched);
               (c) unweighted mean of per-family TPRs.
  threshold    "val": the frozen threshold (99th percentile of validation benign, score > tau);
               "oracle": 99th percentile of the split's own TEST benign scores, so both splits sit
                   at 1% test FPR; this removes threshold transfer.
  F1           weighted F1 with the threshold tuned on weighted validation rows (frozen), and the
               best weighted F1 over all test thresholds (oracle).

Usage: python scripts/posthoc_split_decomposition.py
Writes outputs/paper2_final/stats/posthoc_split_decomposition{,_summary}.csv and
reports/posthoc_split_decomposition.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from leakage_study import load_true_counts  # noqa: E402
from smoke_compare import md  # noqa: E402
from src.models.evaluation import tune_threshold  # noqa: E402
from src.stats.protocol import bootstrap_ci  # noqa: E402

P = ROOT / "outputs/paper2_final/predictions/splits_final_splits/bigflow"
STATS = ROOT / "outputs/paper2_final/stats"
SOURCES = {"ToN_IoT": "ToN-IoT", "CSE_CIC_IDS2018": "CSE-CIC-IDS2018"}
SPLITS = {"random rows": "random_rows", "episodes": "episodes_gap_1_min_embargo_5_min"}
MODELS = ["HistGB", "RandomForest", "MLP"]


def family_weights(test, population):
    """Rescale attack weights so family shares match `population` (dict family -> share)."""
    w = test["design_weight"].to_numpy(float).copy()
    att = test["label"].to_numpy() == 1
    fam = test["family"].astype(str).to_numpy()
    total = w[att].sum()
    present = {f: s for f, s in population.items() if (att & (fam == f)).any()}
    norm = sum(present.values())
    for f, share in present.items():
        m = att & (fam == f)
        w[m] *= (share / norm) * total / w[m].sum()
    return w


def best_f1(y, s, w):
    p, r, _ = precision_recall_curve(y, s, sample_weight=w)
    return float(np.nanmax(2 * p * r / np.maximum(p + r, 1e-12)))


def evaluate(frame, col, population):
    split = frame["split"].astype(str).to_numpy()
    y, s = frame["label"].to_numpy(), frame[col].to_numpy(float)
    val, test = split == "validation", split == "test"
    tau = {"val": np.quantile(s[val & (y == 0)], 0.99), "oracle": np.quantile(s[test & (y == 0)], 0.99)}
    t = frame[test]
    yt, st = y[test], s[test]
    weights = {"pooled": t["design_weight"].to_numpy(float), "reweighted": family_weights(t, population)}
    fam = t["family"].astype(str).to_numpy()
    out = {}
    for tname, thr in tau.items():
        alert = st > thr
        out[f"FPR_{tname}"] = alert[yt == 0].mean()
        for wname, w in weights.items():
            out[f"TPR_{wname}_{tname}"] = np.average(alert[yt == 1], weights=w[yt == 1])
        fams = sorted(set(fam[yt == 1]))
        out[f"TPR_unweighted_family_mean_{tname}"] = np.mean([alert[(yt == 1) & (fam == f)].mean() for f in fams])
    f1_thr = tune_threshold(y[val], s[val], sample_weight=frame["design_weight"].to_numpy(float)[val])
    for wname, w in weights.items():
        out[f"F1_{wname}_val"] = f1_score(yt, (st >= f1_thr).astype(int), sample_weight=w, zero_division=0)
        out[f"F1_{wname}_oracle"] = best_f1(yt, st, w)
    return out


def decomposition_orders(d):
    """
    Split the random-minus-episode gap into family mix, threshold transfer and remainder in both
    orders (mix first; threshold first) and their average. Per seed, then mean with bootstrap CI.
    """
    rows = []
    for (source, model), g in d.groupby(["source", "model"]):
        r = g[g.split == "random rows"].set_index("seed")
        e = g[g.split == "episodes"].set_index("seed")
        for metric in ("TPR", "F1"):
            gap = {k: (r[f"{metric}_{k}"] - e[f"{metric}_{k}"]).to_numpy(float)
                   for k in ("pooled_val", "reweighted_val", "pooled_oracle", "reweighted_oracle")}
            terms = {
                "total": gap["pooled_val"],
                "mix (mix first)": gap["pooled_val"] - gap["reweighted_val"],
                "threshold (mix first)": gap["reweighted_val"] - gap["reweighted_oracle"],
                "threshold (threshold first)": gap["pooled_val"] - gap["pooled_oracle"],
                "mix (threshold first)": gap["pooled_oracle"] - gap["reweighted_oracle"],
                "remainder": gap["reweighted_oracle"],
            }
            terms["mix (average)"] = (terms["mix (mix first)"] + terms["mix (threshold first)"]) / 2
            terms["threshold (average)"] = (terms["threshold (mix first)"] + terms["threshold (threshold first)"]) / 2
            for term, v in terms.items():
                lo, hi = bootstrap_ci(v)
                rows.append({"source": source, "model": model, "metric": metric, "term": term,
                             "mean": v.mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main():
    counts = load_true_counts()
    rows = []
    for folder, source in SOURCES.items():
        att = {a: n for (s, a), n in counts.items() if s == source and a != "Benign"}
        tot = sum(att.values())
        population = {a: n / tot for a, n in att.items()}
        for seed in range(10):
            for split, tag in SPLITS.items():
                frame = pd.read_parquet(P / folder / f"seed{seed}_{tag}.parquet")
                for model in MODELS:
                    rows.append({"source": source, "seed": seed, "split": split, "model": model,
                                 **evaluate(frame, f"{model}|behaviour", population)})
    d = pd.DataFrame(rows)
    d.to_csv(STATS / "posthoc_split_decomposition.csv", index=False)

    metrics = [c for c in d.columns if c.startswith(("TPR_", "F1_", "FPR_"))]
    summ = []
    for (source, model), g in d.groupby(["source", "model"]):
        r = g[g.split == "random rows"].set_index("seed")
        e = g[g.split == "episodes"].set_index("seed")
        for m in metrics:
            diff = (r[m] - e[m]).to_numpy(float)
            lo, hi = bootstrap_ci(diff)
            summ.append({"source": source, "model": model, "metric": m, "random_rows": r[m].mean(),
                         "episodes": e[m].mean(), "gap": diff.mean(), "gap_ci_low": lo, "gap_ci_high": hi,
                         "seeds_random_higher": int((diff > 0).sum())})
    s = pd.DataFrame(summ)
    s.to_csv(STATS / "posthoc_split_decomposition_summary.csv", index=False)
    order = ["TPR_pooled_val", "TPR_reweighted_val", "TPR_unweighted_family_mean_val", "TPR_pooled_oracle",
             "TPR_reweighted_oracle", "TPR_unweighted_family_mean_oracle", "FPR_val", "FPR_oracle",
             "F1_pooled_val", "F1_reweighted_val", "F1_pooled_oracle", "F1_reweighted_oracle"]
    s["metric"] = pd.Categorical(s["metric"], order)
    s = s.sort_values(["source", "model", "metric"])
    text = ["# Post-hoc: decomposition of the split gap (claim C4) — DEVIATION\n", __doc__.strip() + "\n",
            "Gap = random rows − episode split, mean over 10 paired seeds, bootstrap 95% CI.\n",
            md(s[["source", "model", "metric", "random_rows", "episodes", "gap", "gap_ci_low", "gap_ci_high",
                  "seeds_random_higher"]], 3)]
    orders = decomposition_orders(d)
    orders.to_csv(STATS / "posthoc_split_decomposition_orders.csv", index=False)
    wide = orders.pivot_table(index=["source", "model", "metric"], columns="term", values="mean").reset_index()
    cols = ["source", "model", "metric", "total", "mix (mix first)", "threshold (mix first)",
            "threshold (threshold first)", "mix (threshold first)", "mix (average)", "threshold (average)",
            "remainder"]
    text += ["\n## Order of the corrections\n",
             "The two corrections do not commute, so the family-mix and threshold terms are given in both orders "
             "and averaged (the remainder is the same in both). F1 terms are in F1 units (x100 for points).\n",
             md(wide[cols], 3)]
    (ROOT / "outputs/paper2_final/reports/posthoc_split_decomposition.md").write_text("\n".join(text) + "\n")
    print("\n".join(text[-1:]))


if __name__ == "__main__":
    main()
