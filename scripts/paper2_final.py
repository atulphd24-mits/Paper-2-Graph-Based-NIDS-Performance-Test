"""
Paper 2 final runs (Phase 1): leakage-fixed experiments.

Fixes applied (outputs/paper2_final/reports/phase0_code_audit.md):
  A  past-only host features (data/processed/*_working_sample_causal.parquet)
  B  inductive neighbourhood context (scaler, k-NN index, distances, LOF on training rows)
  C  inductive E-GraphSAGE mode (transductive kept for comparison)
  4  family-gated detector: validation split into an expert-choice half and a
     threshold half (disjoint by episode), reported next to the reused version
  5  episodes built from full-data timestamps

Split protocols (per source dataset):
  episodes      attack episodes (gap G) mapped from full-data bounds, families with
                < 3 episodes fall back to a chronological split, embargo E around
                every boundary between splits (main: G = 1 min, E = 5 min)
  chronological every class split by time (reference protocol), same embargo
  random        random rows, stratified by class

Every result row is evaluated on three test subsets with the same thresholds:
  cross-episode  benign test flows + attack test flows from real episodes of families
                 with >= 3 test episodes (headline)
  fallback       benign test flows + attack test flows from the chronological fallback
  all            every test flow

Usage:
  python scripts/paper2_final.py splits     --dataset bigflow --seeds 0-9
  python scripts/paper2_final.py context    --dataset bigflow --seeds 0-9 --protocol episodes --gap 1 --embargo 5 --models HistGB RandomForest MLP
  python scripts/paper2_final.py gated      --dataset bigflow --seeds 0-9
  python scripts/paper2_final.py egraphsage --dataset bigflow --seeds 0-9 --modes inductive transductive
  python scripts/paper2_final.py smoke      --dataset bigflow
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from leakage_study import (NEIGHBOUR_CONTEXT_COLUMNS, load_true_counts, operational_metrics,  # noqa: E402
                           random_split, row_weights,
                           check_benign_weights_constant)
from robustness_study import fit_scores  # noqa: E402
from egraphsage_study import edge_features  # noqa: E402
from src.data import external  # noqa: E402
from src.data.bigflow import behaviour_columns  # noqa: E402
from src.graph.neighborhood_features import NeighborhoodFeatureExtractor  # noqa: E402
from src.graph.preprocessing import FlowFeatureTransformer  # noqa: E402
from src.models.egraphsage import build_endpoint_graph, train_egraphsage  # noqa: E402
from src.models.evaluation import (apply_embargo, chronological_split, episode_split,  # noqa: E402
                                   map_episodes, prevalence_weights, split_validation)
from src.stats.protocol import episode_bootstrap_tpr  # noqa: E402
from src.models.two_branch import FamilyGatedDetector  # noqa: E402
from src.models.predictions import prediction_frame, slug, write_predictions  # noqa: E402

OUT = ROOT / "outputs/paper2_final"
PROCESSED = ROOT / "data/processed"
MAX_ROWS, K = 60_000, 20
REPS = ["flow", "flow + neighbourhood", "flow + host", "flow + neighbourhood + host"]
MIN_TEST_EPISODES = 3
MIN_TRAIN_FLOWS = 500
MIN_VALIDATION_FLOWS = 50

DATASETS = {
    "bigflow": {"sample": "bigflow_working_sample_strat_causal.parquet", "bounds": "episode_bounds_bigflow.parquet",
                "sources": ["ToN-IoT", "CSE-CIC-IDS2018"], "nbr_columns": NEIGHBOUR_CONTEXT_COLUMNS},
    "unsw": {"sample": "unsw_working_sample_strat_causal.parquet", "bounds": "episode_bounds_unsw.parquet",
             "sources": ["UNSW-NB15 (original)"], "nbr_columns": ["sbytes", "dbytes", "spkts", "dpkts", "sttl", "dur"]},
    "ciciot": {"sample": "ciciot_working_sample.parquet", "bounds": None,
               "sources": ["CIC-IoT2023"], "nbr_columns": ["Tot sum", "Rate", "Header_Length", "flow_duration", "Tot size", "IAT"]},
}


def log(message, start=time.time()):
    print(f"[{time.time() - start:7.0f}s] {message}", flush=True)


def parse_seeds(items):
    seeds = []
    for item in items:
        a, _, b = str(item).partition("-")
        seeds += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return seeds


_GIT_AT_START = {}


def git_state():
    """Commit and dirty flag, read once per process (at the run's start) and reused, so a commit
    made while a run is in progress is never recorded as the code the run used."""
    if not _GIT_AT_START:
        git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
        _GIT_AT_START.update(commit=git("rev-parse", "HEAD"),
                             dirty=bool(git("status", "--porcelain", "--untracked-files=no")))
    return _GIT_AT_START


def run_meta(name, args, start, status):
    import sklearn, torch, scipy  # noqa: E401
    meta = {"run": name, "status": status, "args": {k: v for k, v in vars(args).items()},
            "git_commit": git_state()["commit"], "git_dirty_tracked": git_state()["dirty"],
            "started": datetime.fromtimestamp(start).isoformat(), "finished": datetime.now().isoformat(),
            "wall_clock_s": round(time.time() - start, 1), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "torch": torch.__version__, "scipy": scipy.__version__, "host": platform.node()}
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def append(rows, path):
    """
    Append metric rows, keeping columns aligned with the file's header. Column sets differ between
    batches (per-family TPR columns per source, bootstrap columns only for episode protocols);
    when a batch brings new columns the file is rewritten with the widened header, all existing
    rows kept unchanged.
    """
    if not rows:
        return
    new = pd.DataFrame(rows)
    if not path.exists():
        new.to_csv(path, index=False)
        return
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if set(new.columns) <= set(header):
        new.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)
        return
    old = pd.read_csv(path)
    columns = header + [c for c in new.columns if c not in header]
    tmp = path.with_suffix(".tmp")
    pd.concat([old, new], ignore_index=True).reindex(columns=columns).to_csv(tmp, index=False)
    tmp.replace(path)


# ----------------------------------------------------------------------
# Data and splits
# ----------------------------------------------------------------------

# Sampling bookkeeping: never model inputs (design_weight and episode_full identify the episode).
SAMPLING_COLUMNS = ("file_id", "file_row", "design_weight", "episode_full")


def model_features(name, columns):
    """Behaviour (flow) features of a dataset: no identifiers, targets, host or sampling columns."""
    if name == "bigflow":
        features = [c for c in behaviour_columns(columns) if not c.startswith("host_")]
    elif name == "unsw":
        features = external.unsw_behaviour_columns()
    else:
        features = external.ciciot_behaviour_columns(columns)
    return [c for c in features if c not in SAMPLING_COLUMNS]


def sample_info(data):
    """Recorded in every run's meta file: which sample and which inputs the run used."""
    return {"file": data.cfg["sample"], "rows": int(len(data.df)),
            "episode_stratified": "design_weight" in data.df.columns,
            "n_features": len(data.features), "n_host_features": len(data.host_cols)}


class Data:
    def __init__(self, name, sample=None):
        self.name = name
        cfg = dict(DATASETS[name])
        if sample:
            cfg["sample"] = sample
        self.cfg = cfg
        self.df = pd.read_parquet(PROCESSED / cfg["sample"])
        self.features = model_features(name, self.df.columns)
        self.host_cols = [c for c in self.df.columns if c.startswith("host_")]
        self.bounds = pd.read_parquet(PROCESSED / cfg["bounds"]) if cfg["bounds"] else None
        if name == "bigflow":
            self.true_counts = load_true_counts()
        else:
            tag = "unsw" if name == "unsw" else "ciciot"
            counts = json.loads((ROOT / f"outputs/reports/external/{tag}_class_counts.json").read_text())
            self.true_counts = {(cfg["sources"][0], a): n for a, n in counts.items()}

    subsample = True

    def source(self, source, subsample=None):
        subsample = self.subsample if subsample is None else subsample
        sdf = self.df[self.df["source"] == source]
        if subsample and len(sdf) > MAX_ROWS:  # same rows and random_state as the v1 runs
            sdf = sdf.groupby("attack", group_keys=False).sample(frac=MAX_ROWS / len(sdf), random_state=0)
        if "flow_start_milliseconds" in sdf:
            sdf = sdf.sort_values("flow_start_milliseconds", kind="stable")
        return sdf.reset_index(drop=True)

    def make_split(self, sdf, seed, protocol, gap_min=1, embargo_min=5):
        """Returns split, episode ids, mode ('episode' | 'chronological' | 'random')."""
        n = len(sdf)
        attack = sdf["attack"].to_numpy()
        if protocol == "random":
            return random_split(n, attack, seed), np.full(n, None, object), np.full(n, "random", object)
        t = sdf["flow_start_milliseconds"].to_numpy().astype(np.int64)
        strata = (sdf["source"] + "|" + sdf["attack"]).to_numpy()
        if protocol == "chronological":
            split = chronological_split(t, strata)
            episode = np.full(n, None, object)
            mode = np.full(n, "chronological", object)
            ranges = None
        else:
            b = self.bounds[self.bounds["gap_min"] == gap_min]
            episodes = map_episodes(t, strata, b)
            self.assignment_log = []
            split, episode, mode = episode_split(t, sdf["source"], attack, gap_ms=gap_min * 60_000,
                                                 random_state=seed, episodes=episodes,
                                                 assignment_log=self.assignment_log)
            ids = b["key"] + "|" + b["episode"].astype(str)
            ranges = dict(zip(ids, zip(b["start_ms"], b["end_ms"])))
        ips = (sdf["ipv4_src_addr"].to_numpy(), sdf["ipv4_dst_addr"].to_numpy()) \
            if "ipv4_src_addr" in sdf else (None, None)
        split = apply_embargo(split, episode, mode, strata, t, embargo_min * 60_000,
                              episode_ranges=ranges, source=sdf["source"].to_numpy(),
                              src_ip=ips[0], dst_ip=ips[1])
        return split, episode, mode


def test_subsets(split, episode, mode, attack, label):
    """Boolean masks over all rows for the three test subsets, plus bookkeeping."""
    test = split == "test"
    benign = label == 0
    cross_attack = test & ~benign & (mode == "episode")
    eps_per_family = pd.Series(episode[cross_attack]).groupby(attack[cross_attack]).nunique()
    train_flows = pd.Series(attack[(split == "train") & ~benign]).value_counts()
    enough_train = set(train_flows[train_flows >= MIN_TRAIN_FLOWS].index)
    eligible = set(eps_per_family[eps_per_family >= MIN_TEST_EPISODES].index) & enough_train
    dropped = sorted(set(attack[test & ~benign]) - eligible)
    cross = test & (benign | (cross_attack & np.isin(attack, list(eligible))))
    fallback = test & (benign | (mode == "chronological"))
    if (mode == "random").all():
        return {"all": test}, {"dropped_from_cross": [], "cross_families": []}
    return ({"cross-episode": cross, "fallback": fallback, "all": test},
            {"dropped_from_cross": dropped, "cross_families": sorted(eligible),
             "test_episodes_per_family": eps_per_family.to_dict(),
             "low_training_flows": sorted(set(eps_per_family.index) - enough_train)})


def evaluate(y, scores, split, sdf, counts, name, subsets, episode, extra):
    rows = []
    test_attacks = (split == "test") & (y == 1)
    for subset, mask in subsets.items():
        s2 = np.where((split == "test") & ~mask, "excluded", split)
        if not ((s2 == "test") & (y == 1)).any():
            continue
        m = operational_metrics(y, scores, s2, sdf, counts, name)
        att = mask & (y == 1)
        m.update(extra, subset=subset, test_flows=int(mask.sum()),
                 share_of_test_attacks=float(att.sum() / max(test_attacks.sum(), 1)),
                 test_episodes=int(pd.Series(episode[att]).nunique()))
        if subset == "cross-episode":
            # Episode-level cluster bootstrap of TPR@1%FPR (episodes resampled, n = episodes).
            threshold = np.quantile(scores[(split == "validation") & (y == 0)], 0.99)
            weights = row_weights(att, sdf, counts)
            tpr, lo, hi, n = episode_bootstrap_tpr(scores[att] > threshold, weights, episode[att],
                                                   seed=int(extra.get("seed", 0)))
            m.update({"TPR@FPR1pct_episode_boot": tpr, "TPR@FPR1pct_ci_low": lo,
                      "TPR@FPR1pct_ci_high": hi, "n_episodes": n})
        rows.append(m)
    return rows


def blocks(sdf, data, split, with_host=True):
    y = sdf["label"].to_numpy().astype(int)
    train = split == "train"
    X = sdf[data.features].to_numpy(float)
    Xg = FlowFeatureTransformer(numeric_columns=data.features, categorical_columns=[], flag_columns=[]
                                ).fit_transform(sdf, fit_rows=train) if data.name != "bigflow" else \
        FlowFeatureTransformer(numeric_columns=data.features).fit_transform(sdf, fit_rows=train)
    N = NeighborhoodFeatureExtractor(k=K, inductive=True).extract(
        Xg, y, train, sdf["attack"].to_numpy(), raw=sdf, raw_columns=data.cfg["nbr_columns"]).to_numpy()
    out = {"flow": X, "flow + neighbourhood": np.hstack([X, N])}
    if with_host and data.host_cols:
        H = sdf[data.host_cols].to_numpy(float)
        out["flow + host"] = np.hstack([X, H])
        out["flow + neighbourhood + host"] = np.hstack([X, N, H])
    return out


def protocol_label(args):
    if args.protocol == "episodes":
        return f"episodes gap {args.gap} min, embargo {args.embargo} min"
    if args.protocol == "chronological":
        return f"chronological, embargo {args.embargo} min"
    return "random rows"


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def save_predictions(args, data, experiment, source, seed, label, sdf, split, episode, mode, subsets,
                     scores, gap=None, variant=None):
    """Per-flow validation + test scores of one run (secondary response-time metrics)."""
    bounds = (data.bounds[data.bounds["gap_min"] == gap]
              if gap and data.bounds is not None else None)
    weights = row_weights(np.ones(len(sdf), bool), sdf, data.true_counts)
    name = f"seed{seed}_{slug(label)}" + (f"_{slug(variant)}" if variant else "") + ".parquet"
    path = OUT / "predictions" / f"{experiment}{args.tag}" / data.name / slug(source) / name
    write_predictions(path, prediction_frame(sdf, split, episode, mode, subsets, scores, weights, bounds))


def run_splits(args, data):
    """Split-inflation comparison on each source's full working sample."""
    path = OUT / "results" / f"splits_raw{args.tag}.csv"
    protocols = [("random", 0, 0), ("episodes", args.gap, args.embargo), ("chronological", 0, args.embargo)]
    for source in args.sources:
        sdf = data.source(source, subsample=False)
        y = sdf["label"].to_numpy().astype(int)
        X = sdf[data.features].to_numpy(float)
        T = sdf[["flow_start_milliseconds"]].to_numpy(float) if "flow_start_milliseconds" in sdf else None
        for seed in args.seeds:
            for protocol, gap, emb in protocols:
                if protocol != "random" and T is None:
                    continue
                split, episode, mode = data.make_split(sdf, seed, protocol, gap, emb)
                subsets, info = test_subsets(split, episode, mode, sdf["attack"].to_numpy(), y)
                label = {"random": "random rows", "episodes": f"episodes gap {gap} min, embargo {emb} min",
                         "chronological": f"chronological, embargo {emb} min"}[protocol]
                rows, preds = [], {}
                inputs = [("behaviour", X, args.models)] + ([("timestamp only", T, ["HistGB"])] if T is not None else [])
                for rep, M, models in inputs:
                    for model in models:
                        scores = fit_scores(model, M, y, split, seed)
                        preds[f"{model}|{rep}"] = scores
                        rows += evaluate(y, scores, split, sdf, data.true_counts, rep, subsets, episode,
                                         dict(dataset=data.name, source=source, seed=seed, model=model,
                                              representation=rep, protocol=label,
                                              purged=int((split == "purged").sum()),
                                              dropped_from_cross=";".join(info["dropped_from_cross"])))
                save_predictions(args, data, "splits", source, seed, label, sdf, split, episode, mode, subsets,
                                 preds, gap=gap if protocol == "episodes" else None)
                append(rows, path)
            log(f"splits {source} seed {seed} done")


def run_context(args, data):
    path = OUT / "results" / f"context_raw{args.tag}.csv"
    for source in args.sources:
        sdf = data.source(source)
        y = sdf["label"].to_numpy().astype(int)
        for seed in args.seeds:
            split, episode, mode = data.make_split(sdf, seed, args.protocol, args.gap, args.embargo)
            subsets, info = test_subsets(split, episode, mode, sdf["attack"].to_numpy(), y)
            feats = blocks(sdf, data, split)
            rows, preds = [], {}
            for model in args.models:
                for rep in [r for r in args.reps if r in feats]:
                    scores = fit_scores(model, feats[rep], y, split, seed)
                    preds[f"{model}|{rep}"] = scores
                    rows += evaluate(y, scores, split, sdf, data.true_counts, rep, subsets, episode,
                                     dict(dataset=data.name, source=source, seed=seed, model=model,
                                          representation=rep, protocol=protocol_label(args),
                                          purged=int((split == "purged").sum()),
                                          dropped_from_cross=";".join(info["dropped_from_cross"])))
            save_predictions(args, data, "context", source, seed, protocol_label(args), sdf, split, episode, mode,
                             subsets, preds, gap=args.gap if args.protocol == "episodes" else None)
            append(rows, path)
            log(f"context {source} {protocol_label(args)} seed {seed} done")


def run_gated(args, data):
    path = OUT / "results" / f"gated_raw{args.tag}.csv"
    for source in args.sources:
        sdf = data.source(source)
        y = sdf["label"].to_numpy().astype(int)
        attack = sdf["attack"].to_numpy()
        t = sdf["flow_start_milliseconds"].to_numpy()
        check_benign_weights_constant(sdf)  # the detector's thresholds are unweighted benign quantiles
        for seed in args.seeds:
            split, episode, mode = data.make_split(sdf, seed, args.protocol, args.gap, args.embargo)
            subsets, info = test_subsets(split, episode, mode, attack, y)
            feats = blocks(sdf, data, split)
            experts = {k: feats[k] for k in ("flow", "flow + neighbourhood", "flow + host")}
            strata = (sdf["source"] + "|" + sdf["attack"]).to_numpy()
            halves = split_validation(split, episode, mode, strata, t, random_state=seed)
            versions = {
                # reused: one validation set chooses experts AND sets thresholds
                "reused validation": (split, split == "validation"),
                # split: one half chooses experts, the other half sets thresholds
                "split validation": (np.where(halves == "validation_threshold", "validation",
                                              np.where(halves == "validation_select", "selection", halves)),
                                     halves == "validation_select"),
            }
            rows = []
            for version, (eval_split, select_mask) in versions.items():
                det = FamilyGatedDetector(target_fpr=0.01, seed=seed).fit(
                    experts, feats["flow"], y, attack, split == "train", select_mask)
                candidates = {n: det.experts_[n].predict_proba(M)[:, 1] for n, M in experts.items()}
                candidates["ensemble (mean of experts)"] = det.ensemble_score(experts)
                candidates["family-gated"] = det.score(experts, feats["flow"])
                save_predictions(args, data, "gated", source, seed, protocol_label(args), sdf, eval_split, episode,
                                 mode, subsets, candidates,
                                 gap=args.gap if args.protocol == "episodes" else None, variant=version)
                for rep, scores in candidates.items():
                    rows += evaluate(y, scores, eval_split, sdf, data.true_counts, rep, subsets, episode,
                                     dict(dataset=data.name, source=source, seed=seed, representation=rep,
                                          validation=version, protocol=protocol_label(args)))
                append([{"source": source, "seed": seed, "validation": version, "family": f, "expert": e,
                         "pooled_default_lt50_validation_flows": f in det.pooled_families_}
                        for f, e in det.choice_.items()], OUT / "results" / f"gated_choices{args.tag}.csv")
            append(rows, path)
            log(f"gated {source} seed {seed} done")


def run_egraphsage(args, data):
    path = OUT / "results" / f"egraphsage_raw{args.tag}.csv"
    for source in args.sources:
        sdf = data.source(source)
        X = sdf[data.features].to_numpy(float)
        y = sdf["label"].to_numpy().astype(int)
        graphs = {e: build_endpoint_graph(sdf, e) for e in args.endpoints}
        for seed in args.seeds:
            for protocol in args.split_protocols:
                split, episode, mode = (data.make_split(sdf, seed, "random") if protocol == "random"
                                        else data.make_split(sdf, seed, "episodes", args.gap, args.embargo))
                subsets, info = test_subsets(split, episode, mode, sdf["attack"].to_numpy(), y)
                label = "random rows" if protocol == "random" else f"episodes gap {args.gap} min, embargo {args.embargo} min"
                train, val = split == "train", split == "validation"
                extra = dict(dataset=data.name, source=source, seed=seed, protocol=label)
                rows, preds = [], {}
                if args.with_baseline:
                    scores = fit_scores("HistGB", X, y, split, seed)
                    preds["HistGB (flow, no graph)"] = scores
                    rows += evaluate(y, scores, split, sdf, data.true_counts, "HistGB (flow, no graph)", subsets,
                                     episode, dict(extra, model="HistGB (flow, no graph)", mode="-"))
                E = edge_features(X, train)
                for gmode in args.modes:
                    for endpoint, (edge_index, n_nodes) in graphs.items():
                        scores = train_egraphsage(edge_index, E, n_nodes, y, train, val, seed=seed,
                                                  threads=args.threads, inductive=(gmode == "inductive"))
                        name = f"E-GraphSAGE ({endpoint})"
                        preds[f"{name}|{gmode}"] = scores
                        rows += evaluate(y, scores, split, sdf, data.true_counts, name, subsets, episode,
                                         dict(extra, model=name, mode=gmode))
                save_predictions(args, data, "egraphsage", source, seed, label, sdf, split, episode, mode, subsets,
                                 preds, gap=None if protocol == "random" else args.gap)
                append(rows, path)
                log(f"egraphsage {source} {label} seed {seed} done")


def run_smoke(args, data):
    """One seed per source at the given protocol: episode bookkeeping, no models."""
    reports = OUT / "reports"
    rows, assign = [], []
    for source in args.sources:
        sdf = data.source(source)
        y = sdf["label"].to_numpy().astype(int)
        attack = sdf["attack"].to_numpy()
        for seed in args.seeds:
            before = data.make_split(sdf, seed, args.protocol, args.gap, 0)[0]
            split, episode, mode = data.make_split(sdf, seed, args.protocol, args.gap, args.embargo)
            methods = {r["stratum"].split("|", 1)[1]: r["method"] for r in getattr(data, "assignment_log", [])}
            subsets, info = test_subsets(split, episode, mode, attack, y)
            for fam in sorted(set(attack)):
                m = attack == fam
                fam_eps = pd.Series(episode[m])
                rec = {"source": source, "seed": seed, "family": fam, "flows": int(m.sum()),
                       "mode": "benign blocks" if fam == "Benign" else
                       ("chronological fallback" if (mode[m] == "chronological").all() else "episodes"),
                       "episodes_in_sample": int(fam_eps.nunique()) if fam != "Benign" else None}
                for s in ("train", "validation", "test", "purged"):
                    rec[s] = int((m & (split == s)).sum())
                    if s != "purged" and fam != "Benign":
                        rec[f"{s}_episodes"] = int(pd.Series(episode[m & (split == s)]).nunique())
                rec["purged_by_embargo"] = int((m & (split == "purged") & (before != "purged")).sum())
                rec["purged_share"] = rec["purged_by_embargo"] / max(rec["flows"], 1)
                method = methods.get(fam, "-")
                rec["assignment"] = ("chronological" if rec["mode"] == "chronological fallback"
                                     else "largest-first: dominant episode always in training"
                                     if method == "largest-first" else method)
                rec["in_cross_subset"] = fam in info["cross_families"]
                rec["excluded_low_train_flows"] = fam in info.get("low_training_flows", [])
                rec["flag_validation_below_50"] = bool(fam != "Benign" and rec["validation"] < MIN_VALIDATION_FLOWS)
                rec["flag_single_cross_test_episode"] = bool(
                    fam != "Benign" and rec.get("test_episodes") == 1 and rec["mode"] == "episodes")
                assign.append(rec)
            test = split == "test"
            for subset, mask in subsets.items():
                att = mask & (y == 1)
                rows.append({"source": source, "seed": seed, "subset": subset,
                             "test_flows": int(mask.sum()), "test_attack_flows": int(att.sum()),
                             "share_of_test_attacks": float(att.sum() / max((test & (y == 1)).sum(), 1)),
                             "test_episodes": int(pd.Series(episode[att]).nunique())})
    pd.DataFrame(assign).to_csv(reports / f"smoke_family_assignment{args.tag}.csv", index=False)
    pd.DataFrame(rows).to_csv(reports / f"smoke_subsets{args.tag}.csv", index=False)
    log("smoke report written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["splits", "context", "gated", "egraphsage", "smoke"])
    parser.add_argument("--dataset", choices=list(DATASETS), default="bigflow")
    parser.add_argument("--seeds", nargs="+", default=["0-9"])
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--protocol", choices=["episodes", "chronological", "random"], default="episodes")
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["HistGB"])
    parser.add_argument("--reps", nargs="+", default=REPS)
    parser.add_argument("--modes", nargs="+", default=["inductive"])
    parser.add_argument("--endpoints", nargs="+", default=["ip", "ip_port", "random"])
    parser.add_argument("--split-protocols", nargs="+", default=["random", "episodes"])
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--sample-file", default=None, help="override the working-sample file name")
    parser.add_argument("--subsample-60k", action="store_true",
                        help="use the 60k-flow subsample per source (default: full working sample, frozen protocol)")
    args = parser.parse_args()
    args.seeds = parse_seeds(args.seeds)
    data = Data(args.dataset, args.sample_file)
    data.subsample = args.subsample_60k
    args.sample_info = sample_info(data)
    args.sources = args.sources or data.cfg["sources"]

    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    name = f"{args.experiment}_{args.dataset}{args.tag}"
    start = time.time()
    run_meta(name, args, start, "running")
    try:
        {"splits": run_splits, "context": run_context, "gated": run_gated,
         "egraphsage": run_egraphsage, "smoke": run_smoke}[args.experiment](args, data)
        run_meta(name, args, start, "finished")
    except Exception:
        run_meta(name, args, start, "failed")
        raise


if __name__ == "__main__":
    main()
