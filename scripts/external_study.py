"""
External validation on UNSW-NB15 (original features) and CIC-IoT2023.

Part A - label ambiguity on ALL rows (exact behaviour-vector duplicates)
Part B - single-feature shortcuts (how well one feature identifies the family)
Part C - split leakage
         UNSW-NB15: random rows vs 10-min blocks vs attack episodes (+ timestamp only),
                    and the official training/testing split
         CIC-IoT2023: random rows only (no timestamps)
Part D - context features (attack-episode split for UNSW-NB15, random split for CIC-IoT2023)
         flow vs + neighbourhood (both) vs + host (UNSW-NB15 only), GB / RF / MLP,
         and E-GraphSAGE (UNSW-NB15 only)

Every part reuses the protocol, metrics and tests of scripts/leakage_study.py.

Usage:
    python scripts/external_study.py --dataset unsw --parts A B C D
    python scripts/external_study.py --dataset ciciot --parts A B C D
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from leakage_study import duplicate_overlap, operational_metrics, paired_tests, random_split, summarise  # noqa: E402
from robustness_study import fit_scores, usable  # noqa: E402
from egraphsage_study import edge_features  # noqa: E402
from src.data import external  # noqa: E402
from src.graph.host_features import HostBehaviorFeatures  # noqa: E402
from src.graph.neighborhood_features import NeighborhoodFeatureExtractor  # noqa: E402
from src.graph.preprocessing import FlowFeatureTransformer  # noqa: E402
from src.models.egraphsage import build_endpoint_graph, train_egraphsage  # noqa: E402
from src.models.evaluation import blocked_time_split, episode_split  # noqa: E402

OUT = PROJECT_ROOT / "outputs/reports/external"
MODELS = ["HistGB", "RandomForest", "MLP"]
ATTACK_CAP, BENIGN_CAP = 20_000, 100_000


def log(message, start=time.time()):
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------

class Dataset:
    """Full-data statistics + capped working sample for one external dataset."""

    def __init__(self, name):
        self.name = name
        self.tag = "unsw" if name == "unsw" else "ciciot"
        self.cache = PROJECT_ROOT / f"data/processed/{self.tag}_working_sample.parquet"
        self.counts_file = OUT / f"{self.tag}_class_counts.json"

    # -- full-data pass ---------------------------------------------------
    def full_hashes(self):
        """(hash, attack) for every row, without keeping all features in memory."""
        if self.name == "unsw":
            df = external.load_unsw()
            cols = external.unsw_behaviour_columns()
            yield df[cols], df["attack"].to_numpy()
            return
        for path in external.ciciot_files():
            df = external.read_ciciot_file(path)
            yield df[external.ciciot_behaviour_columns(df.columns)], df["attack"].to_numpy()

    # -- working sample ---------------------------------------------------
    def sample(self, seed=42):
        if self.cache.exists():
            return pd.read_parquet(self.cache)
        rng = np.random.default_rng(seed)
        counts = self.class_counts()
        rate = {a: min(1.0, (BENIGN_CAP if a == "Benign" else ATTACK_CAP) / n) for a, n in counts.items()}

        if self.name == "unsw":
            df = external.load_unsw()
            log("UNSW: computing host features on all 2.54M records")
            df["_no_flags"] = 0
            host = HostBehaviorFeatures(
                flags_column="_no_flags", bytes_column="sbytes", packets_column="spkts",
                reply_packets_column="dpkts",
            ).transform(df)
            host.columns = [f"host_{c}" for c in host.columns]
            keep = rng.random(len(df)) < df["attack"].map(rate).to_numpy()
            sample = pd.concat([df.loc[keep].drop(columns="_no_flags"), host.loc[keep]], axis=1)
        else:
            parts = []
            for path in external.ciciot_files():
                df = external.read_ciciot_file(path)
                keep = rng.random(len(df)) < df["attack"].map(rate).to_numpy()
                parts.append(df.loc[keep])
            sample = pd.concat(parts, ignore_index=True)

        sample = sample.reset_index(drop=True)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(self.cache, index=False)
        return sample

    def class_counts(self):
        return json.loads(self.counts_file.read_text())

    def true_counts(self):
        source = "UNSW-NB15 (original)" if self.name == "unsw" else "CIC-IoT2023"
        return {(source, a): n for a, n in self.class_counts().items()}

    def features(self, columns):
        if self.name == "unsw":
            return external.unsw_behaviour_columns()
        return external.ciciot_behaviour_columns(
            [c for c in columns if not c.startswith("host_")])


# ----------------------------------------------------------------------
# Part A - label ambiguity
# ----------------------------------------------------------------------

def part_ambiguity(ds):
    log(f"PART A - label ambiguity ({ds.name}, all rows)")
    hashes, attacks = [], []
    for X, attack in ds.full_hashes():
        hashes.append(pd.util.hash_pandas_object(X, index=False).to_numpy())
        attacks.append(attack)
    h = np.concatenate(hashes)
    a = np.concatenate(attacks)
    is_attack = a != "Benign"

    counts = pd.Series(a).value_counts()
    ds.counts_file.parent.mkdir(parents=True, exist_ok=True)
    ds.counts_file.write_text(json.dumps({k: int(v) for k, v in counts.items()}))

    benign_hashes = np.unique(h[~is_attack])
    per_vector = pd.DataFrame({"h": h, "attack": is_attack}).groupby(["h", "attack"]).size().unstack(fill_value=0)
    for col in (False, True):
        if col not in per_vector:
            per_vector[col] = 0
    summary = {
        "dataset": ds.name, "rows": len(h), "attack_share": float(is_attack.mean()),
        "distinct_vectors": int(len(per_vector)),
        "irreducible_error": float(np.minimum(per_vector[True], per_vector[False]).sum() / len(h)),
        "attack_flows_in_ambiguous_vectors": float(np.isin(h[is_attack], benign_hashes).mean()),
        "duplicate_rows": float(1 - len(per_vector) / len(h)),
    }
    families = []
    for attack in sorted(set(a) - {"Benign"}):
        m = a == attack
        families.append({"attack": attack, "flows": int(m.sum()),
                         "distinct_vectors": int(pd.unique(h[m]).size),
                         "share_identical_to_benign": float(np.isin(h[m], benign_hashes).mean())})
    pd.DataFrame([summary]).to_csv(OUT / f"{ds.tag}_ambiguity_summary.csv", index=False)
    fam = pd.DataFrame(families).sort_values("flows", ascending=False)
    fam.to_csv(OUT / f"{ds.tag}_ambiguity_by_family.csv", index=False)
    print(pd.DataFrame([summary]).round(4).to_string(index=False))
    print(fam.round(4).to_string(index=False))


# ----------------------------------------------------------------------
# Part B - single-feature shortcuts
# ----------------------------------------------------------------------

def part_shortcuts(ds, df, features):
    log(f"PART B - single-feature shortcuts ({ds.name})")
    sub = df.sample(min(len(df), 100_000), random_state=0)
    y_family = sub["attack"].to_numpy()
    y_binary = sub["label"].to_numpy()
    tr, te = train_test_split(np.arange(len(sub)), test_size=0.3, random_state=0, stratify=y_family)
    rows = []
    for col in features:
        x = sub[[col]].to_numpy(float)
        fam = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(x[tr], y_family[tr])
        binary = HistGradientBoostingClassifier(max_iter=60, class_weight="balanced",
                                                random_state=0).fit(x[tr], y_binary[tr])
        rows.append({"feature": col,
                     "family_balanced_accuracy": balanced_accuracy_score(y_family[te], fam.predict(x[te])),
                     "binary_balanced_accuracy": balanced_accuracy_score(y_binary[te], binary.predict(x[te]))})
    out = pd.DataFrame(rows).sort_values("binary_balanced_accuracy", ascending=False)
    out.to_csv(OUT / f"{ds.tag}_single_feature.csv", index=False)
    print(out.head(10).round(3).to_string(index=False))


# ----------------------------------------------------------------------
# Part C - split leakage
# ----------------------------------------------------------------------

def part_splits(ds, df, features, seeds):
    log(f"PART C - split leakage ({ds.name})")
    X = df[features].to_numpy(float)
    y = df["label"].to_numpy().astype(int)
    attack = df["attack"].to_numpy()
    counts = ds.true_counts()
    rows = []
    for seed in seeds:
        splits = {"random rows": random_split(len(df), attack, seed)}
        if ds.name == "unsw":
            ts = df["flow_start_milliseconds"].to_numpy()
            splits["10-min time blocks"] = blocked_time_split(ts, attack, 600_000, random_state=seed)
            splits["attack episodes"] = episode_split(ts, df["source"], attack, random_state=seed)[0]
        for split_name, split in splits.items():
            if not usable(y, split):
                continue
            reps = {"behaviour": X}
            if ds.name == "unsw":
                reps["timestamp only"] = df[["flow_start_milliseconds"]].to_numpy(float)
            for rep, M in reps.items():
                scores = fit_scores("HistGB", M, y, split, seed)
                m = operational_metrics(y, scores, split, df, counts, rep)
                m.update(dataset=ds.name, split=split_name, seed=seed, representation=rep,
                         dup_overlap=duplicate_overlap(X, split))
                rows.append(m)
        log(f"{ds.name}: split seed {seed} done")

    frame, agg = summarise(rows, ["dataset", "representation", "split"])
    frame.to_csv(OUT / f"{ds.tag}_splits_raw.csv", index=False)
    agg.to_csv(OUT / f"{ds.tag}_splits_summary.csv", index=False)
    print(agg[["representation", "split", "TPR@FPR1pct_mean", "TPR@FPR1pct_std", "F1_w_mean",
               "PR_AUC_w_mean", "dup_overlap_mean"]].round(4).to_string(index=False))

    if ds.name == "unsw":
        official_split(seeds)


def official_split(seeds):
    log("PART C2 - official UNSW-NB15 training/testing split")
    train, test, feats = external.load_unsw_official_split()
    Xtr, Xte = train[feats].to_numpy(float), test[feats].to_numpy(float)
    htr = pd.util.hash_pandas_object(pd.DataFrame(Xtr), index=False).to_numpy()
    hte = pd.util.hash_pandas_object(pd.DataFrame(Xte), index=False).to_numpy()
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        val = rng.random(len(train)) < 0.15
        both = pd.concat([train.assign(_s=np.where(val, "validation", "train")),
                          test.assign(_s="test")], ignore_index=True)
        both["source"] = "UNSW-NB15 (official split)"
        split = both["_s"].to_numpy()
        X = both[feats].to_numpy(float)
        y = both["label"].to_numpy().astype(int)
        counts = both.groupby(["source", "attack"]).size().to_dict()  # the split's own mix
        scores = fit_scores("HistGB", X, y, split, seed)
        m = operational_metrics(y, scores, split, both, counts, "behaviour")
        m.update(seed=seed, dup_overlap=float(np.isin(hte, htr).mean()))
        rows.append(m)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "unsw_official_split_raw.csv", index=False)
    cols = ["TPR@FPR1pct", "F1_w", "PR_AUC_w", "testFPR@1pct", "dup_overlap"]
    print("official split:", frame[cols].agg(["mean", "std"]).round(4).to_string())


# ----------------------------------------------------------------------
# Part D - context features and E-GraphSAGE
# ----------------------------------------------------------------------

CONTEXT_COLUMNS = {
    "unsw": ["sbytes", "dbytes", "spkts", "dpkts", "sttl", "dur"],
    "ciciot": ["Tot sum", "Rate", "Header_Length", "flow_duration", "Tot size", "IAT"],
}


def part_context(ds, df, features, seeds, max_rows, k):
    log(f"PART D - context features ({ds.name})")
    if len(df) > max_rows:
        df = df.groupby("attack", group_keys=False).sample(frac=max_rows / len(df), random_state=0)
    df = df.reset_index(drop=True)
    host_cols = [c for c in df.columns if c.startswith("host_")]
    X = df[features].to_numpy(float)
    y = df["label"].to_numpy().astype(int)
    attack = df["attack"].to_numpy()
    counts = ds.true_counts()
    graphs = ({m: build_endpoint_graph(df, m) for m in ("ip", "ip_port", "random")}
              if ds.name == "unsw" else {})
    rows = []

    for seed in seeds:
        if ds.name == "unsw":
            split = episode_split(df["flow_start_milliseconds"].to_numpy(), df["source"], attack,
                                  random_state=seed)[0]
        else:
            split = random_split(len(df), attack, seed)
        if not usable(y, split):
            continue
        train = split == "train"
        # Same columns as the flow features: no port or flag encoding added here.
        Xg = FlowFeatureTransformer(numeric_columns=features, categorical_columns=[],
                                    flag_columns=[]).fit_transform(df, fit_rows=train)
        N = NeighborhoodFeatureExtractor(k=k).extract(
            Xg, y, train, attack, raw=df, raw_columns=CONTEXT_COLUMNS[ds.tag]).to_numpy()
        blocks = {"flow": X, "flow + neighbourhood": np.hstack([X, N])}
        if host_cols:
            H = df[host_cols].to_numpy(float)
            blocks["flow + host"] = np.hstack([X, H])
            blocks["flow + neighbourhood + host"] = np.hstack([X, N, H])

        for model in MODELS:
            for rep, M in blocks.items():
                scores = fit_scores(model, M, y, split, seed)
                m = operational_metrics(y, scores, split, df, counts, rep)
                m.update(dataset=ds.name, seed=seed, model=model, representation=rep)
                rows.append(m)

        if graphs:
            E = edge_features(X, train)
            for mode, (edge_index, n_nodes) in graphs.items():
                scores = train_egraphsage(edge_index, E, n_nodes, y, train, split == "validation",
                                          seed=seed, threads=8)
                m = operational_metrics(y, scores, split, df, counts, "flow")
                m.update(dataset=ds.name, seed=seed, model=f"E-GraphSAGE ({mode})", representation="flow")
                rows.append(m)
        log(f"{ds.name}: context seed {seed} done ({len(df):,} flows)")

    frame, agg = summarise(rows, ["dataset", "model", "representation"])
    frame.to_csv(OUT / f"{ds.tag}_context_raw.csv", index=False)
    agg.to_csv(OUT / f"{ds.tag}_context_summary.csv", index=False)
    print(agg[["model", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std", "PR_AUC_w_mean",
               "F1_w_mean"]].round(4).to_string(index=False))

    tests = [paired_tests(frame[frame.model.isin(MODELS)], ["dataset", "model", "representation"], "flow")]
    if graphs:
        graph_frame = frame[(frame.representation == "flow")
                            & ((frame.model == "HistGB") | frame.model.str.startswith("E-"))]
        tests.append(paired_tests(graph_frame, ["dataset", "model"], "HistGB"))
    tests = pd.concat(tests, ignore_index=True)
    tests.to_csv(OUT / f"{ds.tag}_context_significance.csv", index=False)
    print(tests.round(4).to_string(index=False))


def main(args):
    pd.set_option("display.width", 250)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = Dataset(args.dataset)
    if "A" in args.parts or not ds.counts_file.exists():
        part_ambiguity(ds)
    if set(args.parts) & {"B", "C", "D"}:
        df = ds.sample()
        features = ds.features(df.columns)
        log(f"{ds.name}: working sample {df.shape}, {len(features)} behaviour features")
        if "B" in args.parts:
            part_shortcuts(ds, df, features)
        if "C" in args.parts:
            part_splits(ds, df, features, args.split_seeds)
        if "D" in args.parts:
            part_context(ds, df, features, args.seeds, args.max_rows, args.k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["unsw", "ciciot"], required=True)
    parser.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-rows", type=int, default=60_000)
    parser.add_argument("--k", type=int, default=20)
    main(parser.parse_args())
