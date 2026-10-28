"""
Leakage study on BigFlow (findings 1, 3 and 4).

Part 1 - shortcut features
    * Can behaviour features identify the SOURCE dataset?
    * How much of a pooled detector's score is explained by the source
      prior or the timestamp alone?

Part 3 - split strategy
    Per source: random rows vs 10-minute time blocks vs attack episodes,
    plus the share of test flows whose exact feature vector occurs in
    training (duplicate overlap).

Part 4 - graph and host features under the episode split
    Per source, several seeds: flow features vs + structural graph,
    + neighbourhood, + host behaviour.

Requires data/processed/bigflow_working_sample.parquet
(scripts/build_working_sample.py).

Usage:
    python scripts/leakage_study.py --parts 3 4 --seeds 0 1 2 3 4 \
        --part4-seeds 0 1 2 3 4 5 6 7 8 9

Headline metrics: detection rate at 1% / 0.1% false-positive rate (alert
threshold from validation benign traffic) and PR-AUC with test flows
re-weighted to each source's real attack prevalence.
"""

import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bigflow import behaviour_columns  # noqa: E402
from src.graph.edge_builder import HybridEdgeBuilder  # noqa: E402
from src.graph.neighborhood_features import NeighborhoodFeatureExtractor  # noqa: E402
from src.graph.preprocessing import FlowFeatureTransformer  # noqa: E402
from src.models.two_branch import FamilyGatedDetector  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.models.evaluation import (  # noqa: E402
    blocked_time_split,
    episode_split,
    evaluate_binary,
    evaluate_detection,
    prevalence_weights,
    tune_threshold,
)

SAMPLE = PROJECT_ROOT / "data/processed/bigflow_working_sample.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/reports/leakage_study"

NEIGHBOUR_CONTEXT_COLUMNS = [
    "in_bytes", "out_bytes", "in_pkts", "tcp_win_max_in", "min_ttl", "longest_flow_pkt",
]


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


def model(seed, multiclass=False):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, class_weight=None if multiclass else "balanced",
        random_state=seed,
    )


def fit_score(X, y, split, seed):
    train, val = split == "train", split == "validation"
    clf = model(seed).fit(X[train], y[train])
    scores = clf.predict_proba(X)[:, 1]
    return scores, tune_threshold(y[val], scores[val])


def random_split(n, strata, seed):
    idx = np.arange(n)
    rest, test = train_test_split(idx, test_size=0.15, stratify=strata, random_state=seed)
    train, val = train_test_split(
        rest, test_size=0.15 / 0.85, stratify=strata[rest], random_state=seed
    )
    split = np.empty(n, dtype=object)
    split[train], split[val], split[test] = "train", "validation", "test"
    return split


def duplicate_overlap(X, split):
    """Share of test rows whose exact feature vector appears in training."""
    hashes = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    return float(np.isin(hashes[split == "test"], hashes[split == "train"]).mean())


def load_true_counts():
    """(source, attack) -> flows in the complete data (from label_ambiguity.py)."""
    families = pd.read_csv(OUTPUT_DIR / "label_ambiguity_by_family.csv")
    sources = pd.read_csv(OUTPUT_DIR / "label_ambiguity_by_source.csv")
    counts = {(r.source, r.attack): r.flows for r in families.itertuples()}
    for r in sources.itertuples():
        counts[(r.source, "Benign")] = round(r.flows * (1 - r.attack_share))
    return counts


def check_benign_weights_constant(sdf):
    """
    Alert thresholds are unweighted quantiles of validation benign scores. That equals the
    design-weighted quantile only if benign design weights are constant within each source
    (PROTOCOL_FREEZE.md, section 7). Fail loudly otherwise.
    """
    if "design_weight" not in sdf.columns:
        return
    benign = sdf[sdf["attack"] == "Benign"]
    spread = benign.groupby("source")["design_weight"].agg(lambda w: w.max() / w.min() - 1)
    bad = spread[spread > 1e-9]
    if len(bad):
        raise AssertionError(f"benign design_weight not constant within source(s): {bad.to_dict()}")


def row_weights(rows, sdf, true_counts):
    """
    Weights that make a set of rows represent the full data: the sample's design
    weights (1 / inclusion probability) when present, else class-level
    prevalence weights (full-data class count / sample class count).
    """
    if "design_weight" in sdf.columns:
        return sdf["design_weight"].to_numpy(float)[rows]
    return prevalence_weights(sdf["source"].to_numpy()[rows], sdf["attack"].to_numpy()[rows], true_counts)


def operational_metrics(y, scores, split, sdf, true_counts, name):
    """
    Prevalence-weighted metrics on the test split.

    The F1 threshold is tuned on weighted validation rows; alert thresholds for
    fixed false-positive rates come from validation benign. Weights are the
    sample's design weights when available (see ``row_weights``).
    """
    check_benign_weights_constant(sdf)
    test, val = split == "test", split == "validation"
    attack = sdf["attack"].to_numpy()
    threshold = tune_threshold(y[val], scores[val], sample_weight=row_weights(val, sdf, true_counts))

    return evaluate_detection(
        y[test], scores[test], scores[val & (y == 0)],
        families=attack[test],
        weights=row_weights(test, sdf, true_counts),
        f1_threshold=threshold, name=name,
    )


HEADLINE = ["TPR@FPR1pct", "TPR@FPR0.1pct", "PR_AUC_w", "F1_w", "ROC_AUC"]


def paired_tests(frame, keys, baseline):
    """Wilcoxon signed-rank test of each representation against the baseline, paired by seed."""
    rows = []
    for group, g in frame.groupby(keys[:-1]):
        base = g[g[keys[-1]] == baseline].set_index("seed")
        for rep, r in g.groupby(keys[-1]):
            if rep == baseline:
                continue
            r = r.set_index("seed")
            seeds = base.index.intersection(r.index)
            for metric in ("TPR@FPR1pct", "PR_AUC_w"):
                diff = r.loc[seeds, metric] - base.loc[seeds, metric]
                p = wilcoxon(diff).pvalue if (diff != 0).any() and len(diff) >= 5 else np.nan
                rows.append({
                    **dict(zip(keys[:-1], group if isinstance(group, tuple) else (group,))),
                    keys[-1]: rep, "metric": metric, "mean_diff": diff.mean(),
                    "wins": int((diff > 0).sum()), "seeds": len(diff), "wilcoxon_p": p,
                })
    return pd.DataFrame(rows)


def summarise(rows, keys):
    frame = pd.DataFrame(rows)
    metrics = [m for m in HEADLINE if m in frame.columns]
    extra = [c for c in frame.columns
             if c.startswith(("TPR@FPR1pct_", "testFPR@")) or c == "dup_overlap"]
    agg = frame.groupby(keys)[metrics + extra].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return frame, agg.reset_index()


# ----------------------------------------------------------------------
# Part 1
# ----------------------------------------------------------------------

def part_shortcuts(df, features, seeds):
    log("PART 1 - shortcut features")
    rows = []
    X = df[features].to_numpy(float)
    y = df["label"].to_numpy().astype(int)
    source = df["source"].to_numpy()
    source_onehot = pd.get_dummies(df["source"]).to_numpy(float)
    time_only = df[["flow_start_milliseconds"]].to_numpy(float)

    for seed in seeds:
        split = random_split(len(df), df["source"] + "|" + df["attack"], seed)
        train, test = split == "train", split == "test"

        clf = model(seed, multiclass=True).fit(X[train], source[train])
        rows.append({
            "seed": seed, "task": "identify source from behaviour",
            "balanced_accuracy": balanced_accuracy_score(source[test], clf.predict(X[test])),
        })

        for name, M in (
            ("source prior only", source_onehot),
            ("timestamp only", time_only),
            ("behaviour features", X),
        ):
            scores, threshold = fit_score(M, y, split, seed)
            metrics = evaluate_binary(y[test], scores[test], threshold, name=name)
            metrics.update(seed=seed, task="pooled attack detection")
            rows.append(metrics)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "part1_shortcuts_raw.csv", index=False)
    source_id = frame[frame["task"] == "identify source from behaviour"]["balanced_accuracy"]
    detection = frame[frame["task"] == "pooled attack detection"].groupby("Model")[
        ["F1", "PR_AUC", "ROC_AUC"]].agg(["mean", "std"])
    print(f"\nSource identification from behaviour: balanced accuracy "
          f"{source_id.mean():.4f} ± {source_id.std():.4f}")
    print("\nPooled detection (random split):\n", detection.round(4).to_string())

    # Which single features give the source away?
    per_feature = []
    sub = df.sample(min(len(df), 100_000), random_state=0)
    for col in features:
        clf = model(0, multiclass=True).set_params(max_iter=50)
        Xc = sub[[col]].to_numpy(float)
        tr, te = train_test_split(np.arange(len(sub)), test_size=0.3, random_state=0,
                                  stratify=sub["source"])
        clf.fit(Xc[tr], sub["source"].to_numpy()[tr])
        per_feature.append({
            "feature": col,
            "source_balanced_accuracy": balanced_accuracy_score(
                sub["source"].to_numpy()[te], clf.predict(Xc[te])),
        })
    per_feature = pd.DataFrame(per_feature).sort_values("source_balanced_accuracy", ascending=False)
    per_feature.to_csv(OUTPUT_DIR / "part1_single_feature_source_id.csv", index=False)
    print("\nSingle features that best identify the source:\n",
          per_feature.head(10).round(4).to_string(index=False))


# ----------------------------------------------------------------------
# Part 3
# ----------------------------------------------------------------------

def part_splits(df, features, seeds, true_counts):
    log("PART 3 - split strategies")
    rows = []
    for source, sdf in df.groupby("source"):
        sdf = sdf.reset_index(drop=True)
        X = sdf[features].to_numpy(float)
        y = sdf["label"].to_numpy().astype(int)
        ts = sdf["flow_start_milliseconds"].to_numpy()
        attack = sdf["attack"].to_numpy()
        if y.mean() > 0.99 or y.mean() < 0.01:
            log(f"{source}: skipped (one class dominates: attack share {y.mean():.3f})")
            continue

        for seed in seeds:
            splits = {
                "random rows": random_split(len(sdf), attack, seed),
                "10-min time blocks": blocked_time_split(ts, attack, 600_000, random_state=seed),
                "attack episodes": episode_split(ts, sdf["source"], attack, random_state=seed)[0],
            }
            for split_name, split in splits.items():
                if len(np.unique(y[split == "validation"])) < 2 or len(np.unique(y[split == "test"])) < 2:
                    continue
                for rep, M in (("behaviour", X), ("timestamp only", ts[:, None].astype(float))):
                    scores, _ = fit_score(M, y, split, seed)
                    metrics = operational_metrics(y, scores, split, sdf, true_counts, rep)
                    metrics.update(source=source, split=split_name, seed=seed, representation=rep,
                                   dup_overlap=duplicate_overlap(X, split))
                    rows.append(metrics)
            log(f"{source}: seed {seed} done")

    frame, agg = summarise(rows, ["source", "representation", "split"])
    frame.to_csv(OUTPUT_DIR / "part3_splits_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "part3_splits_summary.csv", index=False)
    cols = ["source", "representation", "split", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
            "PR_AUC_w_mean", "F1_w_mean", "dup_overlap_mean"]
    print("\n", agg[cols].round(4).to_string(index=False))


# ----------------------------------------------------------------------
# Part 4
# ----------------------------------------------------------------------

def structural_features(sdf, features, k, fit_rows):
    builder = HybridEdgeBuilder(
        graph=nx.Graph(), dataframe=sdf, feature_columns=features,
        protocol_column="protocol", timestamp_column="flow_start_milliseconds",
        src_port_column="l4_src_port", dst_port_column="l4_dst_port",
        duration_column="flow_duration_milliseconds", bytes_column="in_bytes",
        packets_column="in_pkts", k=k, threshold=0.0,
        transformer=FlowFeatureTransformer(numeric_columns=features), fit_rows=fit_rows,
    )
    builder.prepare_features()
    builder.build_knn()
    builder.build_edges()
    G = builder.graph
    degree = dict(G.degree())
    pagerank = nx.pagerank(G, weight="weight")
    size = {n: len(c) for c in nx.connected_components(G) for n in c}
    nodes = range(len(sdf))
    return np.array([[degree[n], pagerank[n], size[n]] for n in nodes]), builder.features


def part_graph(df, features, seeds, max_rows, k, true_counts, with_structural):
    log("PART 4 - graph and host features under the episode split")
    host_cols = [c for c in df.columns if c.startswith("host_")]
    rows = []

    for source, sdf in df.groupby("source"):
        if sdf["label"].mean() > 0.99 or sdf["label"].mean() < 0.01:
            continue
        if len(sdf) > max_rows:
            sdf = sdf.groupby("attack", group_keys=False).sample(
                frac=max_rows / len(sdf), random_state=0)
        sdf = sdf.sort_values("flow_start_milliseconds").reset_index(drop=True)

        X = sdf[features].to_numpy(float)
        H = sdf[host_cols].to_numpy(float)
        y = sdf["label"].to_numpy().astype(int)
        attack = sdf["attack"].to_numpy()
        ts = sdf["flow_start_milliseconds"].to_numpy()

        for seed in seeds:
            split, _, mode = episode_split(ts, sdf["source"], attack, random_state=seed)
            train, test = split == "train", split == "test"
            if len(np.unique(y[split == "validation"])) < 2 or (y[split == "validation"] == 0).sum() < 100:
                continue

            # The graph itself is label-free; only scaling uses training rows.
            if with_structural:
                structural, X_graph = structural_features(sdf, features, k, train)
            else:
                X_graph = FlowFeatureTransformer(numeric_columns=features).fit_transform(
                    sdf, fit_rows=train)
            neighbourhood = NeighborhoodFeatureExtractor(k=k).extract(
                X_graph, y, train, attack, raw=sdf, raw_columns=NEIGHBOUR_CONTEXT_COLUMNS,
            ).to_numpy()

            blocks = {
                "flow": X,
                "flow + neighbourhood": np.hstack([X, neighbourhood]),
                "flow + host": np.hstack([X, H]),
                "flow + neighbourhood + host": np.hstack([X, neighbourhood, H]),
            }
            if with_structural:
                blocks["flow + structural"] = np.hstack([X, structural])
            for rep, M in blocks.items():
                scores, _ = fit_score(M, y, split, seed)
                metrics = operational_metrics(y, scores, split, sdf, true_counts, rep)
                metrics.update(source=source, seed=seed, representation=rep,
                               chronological_share=float((mode[test] == "chronological").mean()))
                rows.append(metrics)
            log(f"{source}: seed {seed} done ({len(sdf):,} flows)")

    frame, agg = summarise(rows, ["source", "representation"])
    frame.to_csv(OUTPUT_DIR / "part4_graph_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "part4_graph_summary.csv", index=False)
    cols = ["source", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
            "TPR@FPR0.1pct_mean", "PR_AUC_w_mean", "PR_AUC_w_std", "F1_w_mean"]
    print("\n", agg[cols].round(4).to_string(index=False))

    tests = paired_tests(frame, ["source", "representation"], "flow")
    tests.to_csv(OUTPUT_DIR / "part4_graph_significance.csv", index=False)
    print("\nPaired Wilcoxon tests vs flow-only:\n", tests.round(4).to_string(index=False))


# ----------------------------------------------------------------------
# Part 5
# ----------------------------------------------------------------------

def part_two_branch(df, features, seeds, max_rows, k, true_counts, sources):
    log("PART 5 - family-gated two-branch detector")
    host_cols = [c for c in df.columns if c.startswith("host_")]
    rows, choices = [], []

    for source in sources:
        sdf = df[df["source"] == source]
        if len(sdf) > max_rows:
            sdf = sdf.groupby("attack", group_keys=False).sample(
                frac=max_rows / len(sdf), random_state=0)
        sdf = sdf.sort_values("flow_start_milliseconds").reset_index(drop=True)

        X = sdf[features].to_numpy(float)
        H = sdf[host_cols].to_numpy(float)
        y = sdf["label"].to_numpy().astype(int)
        attack = sdf["attack"].to_numpy()
        ts = sdf["flow_start_milliseconds"].to_numpy()

        for seed in seeds:
            split, _, _ = episode_split(ts, sdf["source"], attack, random_state=seed)
            train, val = split == "train", split == "validation"
            if len(np.unique(y[val])) < 2 or (y[val] == 0).sum() < 100:
                continue

            X_graph = FlowFeatureTransformer(numeric_columns=features).fit_transform(
                sdf, fit_rows=train)
            neighbourhood = NeighborhoodFeatureExtractor(k=k).extract(
                X_graph, y, train, attack, raw=sdf, raw_columns=NEIGHBOUR_CONTEXT_COLUMNS,
            ).to_numpy()

            experts = {
                "flow": X,
                "flow + neighbourhood": np.hstack([X, neighbourhood]),
                "flow + host": np.hstack([X, H]),
            }
            detector = FamilyGatedDetector(target_fpr=0.01, seed=seed).fit(
                experts, X, y, attack, train, val)

            candidates = {
                name: detector.experts_[name].predict_proba(M)[:, 1]
                for name, M in experts.items()
            }
            candidates["ensemble (mean of experts)"] = detector.ensemble_score(experts)
            candidates["family-gated (ours)"] = detector.score(experts, X)

            for rep, scores in candidates.items():
                metrics = operational_metrics(y, scores, split, sdf, true_counts, rep)
                metrics.update(source=source, seed=seed, representation=rep)
                rows.append(metrics)

            for fam, expert in detector.choice_.items():
                choices.append({"source": source, "seed": seed, "family": fam, "expert": expert})
            log(f"{source}: seed {seed} done")

    frame, agg = summarise(rows, ["source", "representation"])
    frame.to_csv(OUTPUT_DIR / "part5_two_branch_raw.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "part5_two_branch_summary.csv", index=False)
    choice_table = (
        pd.DataFrame(choices).groupby(["source", "family"])["expert"]
        .value_counts().unstack(fill_value=0)
    )
    choice_table.to_csv(OUTPUT_DIR / "part5_expert_choices.csv")

    cols = ["source", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
            "TPR@FPR0.1pct_mean", "PR_AUC_w_mean", "PR_AUC_w_std", "F1_w_mean", "F1_w_std"]
    print("\n", agg[cols].round(4).to_string(index=False))
    print("\nExpert chosen per family (count over seeds):\n", choice_table.to_string())

    tests = pd.concat([
        paired_tests(frame, ["source", "representation"], baseline)
        .query("representation == 'family-gated (ours)'").assign(baseline=baseline)
        for baseline in ("flow", "flow + host", "ensemble (mean of experts)")
    ])
    tests.to_csv(OUTPUT_DIR / "part5_two_branch_significance.csv", index=False)
    print("\nPaired Wilcoxon tests for the gated detector:\n", tests.round(4).to_string(index=False))


def main(parts, seeds, max_rows, k, part4_seeds, with_structural, sources):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    df = pd.read_parquet(SAMPLE)
    features = [c for c in behaviour_columns(df.columns)
                if not c.startswith("host_") and c not in ("file_id", "file_row")]
    log(f"Working sample {df.shape}, behaviour features: {len(features)}")

    true_counts = load_true_counts()

    if 1 in parts:
        part_shortcuts(df, features, seeds)
    if 3 in parts:
        part_splits(df, features, seeds, true_counts)
    if 4 in parts:
        part_graph(df, features, part4_seeds or seeds, max_rows, k, true_counts, with_structural)
    if 5 in parts:
        part_two_branch(df, features, part4_seeds or seeds, max_rows, k, true_counts, sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-rows", type=int, default=60_000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--part4-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--structural", action="store_true",
                        help="also build the k-NN graph and structural features (slow)")
    parser.add_argument("--sources", nargs="+", default=["ToN-IoT", "CSE-CIC-IDS2018"],
                        help="sources for part 5")
    args = parser.parse_args()
    main(args.parts, args.seeds, args.max_rows, args.k, args.part4_seeds, args.structural,
         args.sources)
