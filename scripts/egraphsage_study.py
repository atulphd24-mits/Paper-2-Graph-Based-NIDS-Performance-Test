"""
E-GraphSAGE under the leakage-safe protocol.

For each source (ToN-IoT, CSE-CIC-IDS2018), split (random rows vs attack
episodes) and seed, trains:

* E-GraphSAGE with three endpoint definitions - "ip", "ip_port" and the
  "random" control (no shared endpoints, so no message passing benefit);
* a HistGB flow-only reference on the same split.

Shows how much a published graph-IDS design gains from random splits, and
whether its graph structure helps once attack episodes are kept apart.

Usage:
    python scripts/egraphsage_study.py [--seeds 0 1 2 3 4] [--max-rows 60000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from leakage_study import (  # noqa: E402
    OUTPUT_DIR,
    SAMPLE,
    load_true_counts,
    operational_metrics,
    paired_tests,
    random_split,
    summarise,
)
from robustness_study import fit_scores, source_frame, usable  # noqa: E402
from src.data.bigflow import behaviour_columns  # noqa: E402
from src.models.egraphsage import build_endpoint_graph, train_egraphsage  # noqa: E402
from src.models.evaluation import episode_split  # noqa: E402

NODE_MODES = ["ip", "ip_port", "random"]


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


def edge_features(X, train):
    X = np.sign(X) * np.log1p(np.abs(np.nan_to_num(X)))
    return StandardScaler().fit(X[train]).transform(X).astype(np.float32)


def main(args):
    pd.set_option("display.width", 250)
    df = pd.read_parquet(SAMPLE)
    features = [c for c in behaviour_columns(df.columns)
                if not c.startswith("host_") and c not in ("file_id", "file_row")]
    true_counts = load_true_counts()
    rows = []

    for source in args.sources:
        sdf = source_frame(df, source, args.max_rows)
        X = sdf[features].to_numpy(float)
        y = sdf["label"].to_numpy().astype(int)
        attack = sdf["attack"].to_numpy()
        ts = sdf["flow_start_milliseconds"].to_numpy()
        graphs = {mode: build_endpoint_graph(sdf, mode) for mode in NODE_MODES}
        log(f"{source}: {len(sdf):,} flows; nodes "
            + ", ".join(f"{m}={n:,}" for m, (_, n) in graphs.items()))

        for seed in args.seeds:
            splits = {
                "random rows": random_split(len(sdf), attack, seed),
                "attack episodes": episode_split(ts, sdf["source"], attack, random_state=seed)[0],
            }
            for split_name, split in splits.items():
                if not usable(y, split):
                    continue
                train, val = split == "train", split == "validation"
                E = edge_features(X, train)

                candidates = {"HistGB (flow, no graph)": fit_scores("HistGB", X, y, split, seed)}
                for mode, (edge_index, num_nodes) in graphs.items():
                    candidates[f"E-GraphSAGE ({mode})"] = train_egraphsage(
                        edge_index, E, num_nodes, y, train, val,
                        max_epochs=args.epochs, seed=seed, threads=args.threads)

                for model_name, scores in candidates.items():
                    m = operational_metrics(y, scores, split, sdf, true_counts, model_name)
                    m.update(source=source, split=split_name, seed=seed, model=model_name)
                    rows.append(m)
                log(f"{source}: seed {seed}, {split_name} done")

    frame, agg = summarise(rows, ["source", "split", "model"])
    frame.to_csv(OUTPUT_DIR / "egraphsage_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "egraphsage_summary.csv", index=False)
    print("\n", agg[["source", "split", "model", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
                     "PR_AUC_w_mean", "F1_w_mean", "F1_w_std"]].round(4).to_string(index=False))

    tests = paired_tests(frame, ["source", "split", "model"], "HistGB (flow, no graph)")
    tests.to_csv(OUTPUT_DIR / "egraphsage_significance.csv", index=False)
    print("\nPaired Wilcoxon tests vs HistGB flow-only:\n", tests.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["ToN-IoT", "CSE-CIC-IDS2018"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-rows", type=int, default=60_000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    main(parser.parse_args())
