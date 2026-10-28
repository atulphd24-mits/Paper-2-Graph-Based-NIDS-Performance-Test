"""
Robustness checks for the leakage study.

Part A - model families
    Repeats finding 3 (random vs episode split) and finding 4 (flow vs
    neighbourhood / host context under the episode split) with Random
    Forest and an MLP, so conclusions do not rest on gradient boosting alone.

Part B - episode gap sensitivity
    Repeats the episode-split results with gaps of 1, 5 and 30 minutes
    (gradient boosting), and reports how many episodes each gap produces.

Uses the same working sample, metrics (detection rate at fixed FPR,
prevalence-weighted PR-AUC / F1) and paired Wilcoxon tests as
scripts/leakage_study.py.

Usage:
    python scripts/robustness_study.py --parts A B
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from leakage_study import (  # noqa: E402
    NEIGHBOUR_CONTEXT_COLUMNS,
    OUTPUT_DIR,
    SAMPLE,
    duplicate_overlap,
    load_true_counts,
    operational_metrics,
    paired_tests,
    random_split,
    summarise,
)
from src.data.bigflow import behaviour_columns  # noqa: E402
from src.graph.neighborhood_features import NeighborhoodFeatureExtractor  # noqa: E402
from src.graph.preprocessing import FlowFeatureTransformer  # noqa: E402
from src.models.evaluation import episode_split  # noqa: E402

SOURCES = ["ToN-IoT", "CSE-CIC-IDS2018"]


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_model(name, seed):
    if name == "HistGB":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            max_features="sqrt", n_jobs=-1, random_state=seed)
    if name == "MLP":
        return make_pipeline(
            FunctionTransformer(signed_log),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512,
                learning_rate_init=1e-3, max_iter=100, early_stopping=True,
                n_iter_no_change=8, random_state=seed),
        )
    raise ValueError(f"unknown model {name}")


def balanced_rows(y, rows, seed):
    """MLPs have no class weights: oversample the minority class in training."""
    rng = np.random.default_rng(seed)
    pos, neg = rows[y[rows] == 1], rows[y[rows] == 0]
    small, large = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = rng.choice(small, len(large) - len(small), replace=True)
    return np.concatenate([rows, extra])


def fit_scores(model_name, X, y, split, seed):
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    train_rows = np.flatnonzero(split == "train")
    if model_name == "MLP":
        train_rows = balanced_rows(y, train_rows, seed)
    clf = make_model(model_name, seed).fit(X[train_rows], y[train_rows])
    return clf.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------
# Shared per-source preparation
# ----------------------------------------------------------------------

def source_frame(df, source, max_rows):
    sdf = df[df["source"] == source]
    if max_rows and len(sdf) > max_rows:
        sdf = sdf.groupby("attack", group_keys=False).sample(
            frac=max_rows / len(sdf), random_state=0)
    return sdf.sort_values("flow_start_milliseconds").reset_index(drop=True)


def context_blocks(sdf, features, host_cols, split, k):
    X = sdf[features].to_numpy(float)
    H = sdf[host_cols].to_numpy(float)
    y = sdf["label"].to_numpy().astype(int)
    train = split == "train"
    X_graph = FlowFeatureTransformer(numeric_columns=features).fit_transform(sdf, fit_rows=train)
    N = NeighborhoodFeatureExtractor(k=k).extract(
        X_graph, y, train, sdf["attack"].to_numpy(),
        raw=sdf, raw_columns=NEIGHBOUR_CONTEXT_COLUMNS,
    ).to_numpy()
    return {
        "flow": X,
        "flow + neighbourhood": np.hstack([X, N]),
        "flow + host": np.hstack([X, H]),
        "flow + neighbourhood + host": np.hstack([X, N, H]),
    }


def usable(y, split):
    val = split == "validation"
    return len(np.unique(y[val])) == 2 and (y[val] == 0).sum() >= 100


# ----------------------------------------------------------------------
# Part A - model families
# ----------------------------------------------------------------------

def part_models(df, features, host_cols, true_counts, models, split_seeds,
                feature_seeds, max_rows, k):
    log("PART A1 - split strategy with other model families")
    rows = []
    for source in SOURCES:
        sdf = source_frame(df, source, max_rows=None)
        X = sdf[features].to_numpy(float)
        y = sdf["label"].to_numpy().astype(int)
        attack = sdf["attack"].to_numpy()
        ts = sdf["flow_start_milliseconds"].to_numpy()
        for seed in split_seeds:
            splits = {
                "random rows": random_split(len(sdf), attack, seed),
                "attack episodes": episode_split(ts, sdf["source"], attack, random_state=seed)[0],
            }
            for split_name, split in splits.items():
                if not usable(y, split):
                    continue
                for model_name in models:
                    scores = fit_scores(model_name, X, y, split, seed)
                    m = operational_metrics(y, scores, split, sdf, true_counts, model_name)
                    m.update(source=source, split=split_name, seed=seed, model=model_name,
                             dup_overlap=duplicate_overlap(X, split))
                    rows.append(m)
            log(f"A1 {source}: seed {seed} done")

    frame, agg = summarise(rows, ["source", "model", "split"])
    frame.to_csv(OUTPUT_DIR / "robustA1_splits_models_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "robustA1_splits_models_summary.csv", index=False)
    print("\n", agg[["source", "model", "split", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
                     "PR_AUC_w_mean", "F1_w_mean", "F1_w_std"]].round(4).to_string(index=False))

    log("PART A2 - context features with other model families")
    rows = []
    for source in SOURCES:
        sdf = source_frame(df, source, max_rows)
        y = sdf["label"].to_numpy().astype(int)
        ts = sdf["flow_start_milliseconds"].to_numpy()
        for seed in feature_seeds:
            split, _, _ = episode_split(ts, sdf["source"], sdf["attack"], random_state=seed)
            if not usable(y, split):
                continue
            blocks = context_blocks(sdf, features, host_cols, split, k)
            for model_name in models:
                for rep, M in blocks.items():
                    scores = fit_scores(model_name, M, y, split, seed)
                    m = operational_metrics(y, scores, split, sdf, true_counts, rep)
                    m.update(source=source, seed=seed, model=model_name, representation=rep)
                    rows.append(m)
            log(f"A2 {source}: seed {seed} done")

    frame, agg = summarise(rows, ["source", "model", "representation"])
    frame.to_csv(OUTPUT_DIR / "robustA2_features_models_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "robustA2_features_models_summary.csv", index=False)
    print("\n", agg[["source", "model", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
                     "PR_AUC_w_mean", "F1_w_mean"]].round(4).to_string(index=False))

    tests = paired_tests(frame, ["source", "model", "representation"], "flow")
    tests.to_csv(OUTPUT_DIR / "robustA2_features_models_significance.csv", index=False)
    print("\nPaired Wilcoxon tests vs flow-only:\n", tests.round(4).to_string(index=False))


# ----------------------------------------------------------------------
# Part B - episode gap sensitivity
# ----------------------------------------------------------------------

def part_gaps(df, features, host_cols, true_counts, gaps_min, seeds, max_rows, k):
    log("PART B - episode gap sensitivity")
    rows, episode_rows = [], []
    for source in SOURCES:
        sdf = source_frame(df, source, max_rows)
        y = sdf["label"].to_numpy().astype(int)
        attack = sdf["attack"].to_numpy()
        ts = sdf["flow_start_milliseconds"].to_numpy()
        X = sdf[features].to_numpy(float)

        for gap in gaps_min:
            for seed in seeds:
                split, episode, mode = episode_split(
                    ts, sdf["source"], attack, gap_ms=gap * 60_000, random_state=seed)
                if seed == seeds[0]:
                    attack_rows = attack != "Benign"
                    counts = pd.Series(episode[attack_rows]).groupby(attack[attack_rows]).nunique()
                    for fam, n in counts.items():
                        episode_rows.append({"source": source, "gap_min": gap,
                                             "family": fam, "episodes": n})
                if not usable(y, split):
                    continue
                blocks = context_blocks(sdf, features, host_cols, split, k)
                for rep in ("flow", "flow + host", "flow + neighbourhood + host"):
                    scores = fit_scores("HistGB", blocks[rep], y, split, seed)
                    m = operational_metrics(y, scores, split, sdf, true_counts, rep)
                    m.update(source=source, gap_min=gap, seed=seed, representation=rep,
                             dup_overlap=duplicate_overlap(X, split),
                             chronological_share=float((mode[split == "test"] == "chronological").mean()))
                    rows.append(m)
                log(f"B {source}: gap {gap} min, seed {seed} done")

    frame, agg = summarise(rows, ["source", "gap_min", "representation"])
    frame.to_csv(OUTPUT_DIR / "robustB_gap_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "robustB_gap_summary.csv", index=False)
    episodes = pd.DataFrame(episode_rows).pivot_table(
        index=["source", "family"], columns="gap_min", values="episodes")
    episodes.to_csv(OUTPUT_DIR / "robustB_gap_episode_counts.csv")

    print("\n", agg[["source", "gap_min", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
                     "PR_AUC_w_mean", "F1_w_mean", "dup_overlap_mean"]].round(4).to_string(index=False))
    print("\nEpisodes per family by gap (minutes):\n", episodes.to_string())

    tests = paired_tests(frame, ["source", "gap_min", "representation"], "flow")
    tests.to_csv(OUTPUT_DIR / "robustB_gap_significance.csv", index=False)
    print("\nPaired Wilcoxon tests vs flow-only:\n", tests.round(4).to_string(index=False))


def main(args):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    df = pd.read_parquet(SAMPLE)
    features = [c for c in behaviour_columns(df.columns)
                if not c.startswith("host_") and c not in ("file_id", "file_row")]
    host_cols = [c for c in df.columns if c.startswith("host_")]
    true_counts = load_true_counts()
    log(f"Working sample {df.shape}")

    if "A" in args.parts:
        part_models(df, features, host_cols, true_counts, args.models,
                    args.split_seeds, args.feature_seeds, args.max_rows, args.k)
    if "B" in args.parts:
        part_gaps(df, features, host_cols, true_counts, args.gaps,
                  args.gap_seeds, args.max_rows, args.k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["A", "B"])
    parser.add_argument("--models", nargs="+", default=["RandomForest", "MLP"])
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--feature-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 5, 30])
    parser.add_argument("--gap-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-rows", type=int, default=60_000)
    parser.add_argument("--k", type=int, default=20)
    main(parser.parse_args())
