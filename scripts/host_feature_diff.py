"""
Diagnostic: old (tumbling-window) vs new (past-only) host features.

For every source (BigFlow's four sources and UNSW-NB15) and every host feature:
  * share of values that changed;
  * median and 95th-percentile relative change, |new - old| / max(|old|, |new|)
    (0 = unchanged, 1 = one of the two is 0 or the sign flipped);
  * Spearman correlation between old and new values;
  * the same statistics separately for benign and attack flows, used to list the
    10 features that changed most for each.
Plus a direct look-ahead measure: for sampled attack flows, the number of flows
of the SAME source host and SAME class that start later but fall in the same old
60-second bin. Such flows are always in the same 1-minute-gap episode (two
same-class flows less than 60 s apart cannot be separated by a gap > 1 min), so
this counts how many future flows of the flow's own attack episode the old
60-second features included.

Outputs:
    outputs/paper2_final/reports/host_feature_diff.csv
    outputs/paper2_final/reports/host_feature_lookahead.csv
    outputs/paper2_final/reports/host_feature_diff_summary.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import external  # noqa: E402
from src.data.bigflow import bigflow_dir, list_files, read_file  # noqa: E402

P = ROOT / "data/processed"
REPORTS = ROOT / "outputs/paper2_final/reports"
BIN = 60_000


def md(df, digits=3):
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, float) else str(v)  # noqa: E731
    return "\n".join([head, sep] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def parse(col):
    name = col.replace("host_", "")
    view_window, stat = name.split("_", 1)
    for w in ("10s", "60s", "1s"):
        if view_window.endswith(w):
            return view_window[: -len(w)], w, stat
    return view_window, "?", stat


def rel_change(old, new):
    denom = np.maximum(np.abs(old), np.abs(new))
    return np.where(denom > 0, np.abs(new - old) / np.where(denom > 0, denom, 1), 0.0)


def feature_rows(dataset, source, old, new, label):
    rows = []
    for col in [c for c in old.columns if c.startswith("host_")]:
        view, window, stat = parse(col)
        for group, mask in (("all", np.ones(len(old), bool)), ("benign", label == 0), ("attack", label == 1)):
            o, n = old.loc[mask, col].to_numpy(float), new.loc[mask, col].to_numpy(float)
            if len(o) == 0:
                continue
            r = rel_change(o, n)
            rho = spearmanr(o, n).statistic if (np.ptp(o) > 0 and np.ptp(n) > 0) else np.nan
            rows.append({"dataset": dataset, "source": source, "flows": group, "feature": col,
                         "view": view, "window": window, "statistic": stat, "n": len(o),
                         "share_changed": float((np.abs(n - o) > 1e-9).mean()),
                         "median_rel_change": float(np.median(r)), "p95_rel_change": float(np.quantile(r, 0.95)),
                         "mean_rel_change": float(r.mean()), "spearman": float(rho)})
    return rows


def lookahead(sample, traffic):
    """Future same-host, same-class flows inside the old 60-s bin, for sampled attack flows."""
    att = sample[sample["label"] == 1]
    out = []
    traffic = traffic.assign(_bin=traffic["t"] // BIN)
    groups = {k: g for k, g in traffic.groupby(["src", "attack"], sort=False)}
    for (src, cls), s in att.groupby(["src", "attack"], sort=False):
        g = groups.get((src, cls))
        times = np.sort(g["t"].to_numpy()) if g is not None else np.array([], np.int64)
        t = s["t"].to_numpy()
        bin_end = (t // BIN + 1) * BIN
        # flows strictly later than t, before the end of t's bin (same-time flows are ignored,
        # which slightly under-counts the look-ahead)
        future = np.searchsorted(times, bin_end, side="left") - np.searchsorted(times, t, side="right")
        out.append(pd.DataFrame({"attack": cls, "future_same_episode": future}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["attack", "future_same_episode"])


def bigflow():
    old = pd.read_parquet(P / "bigflow_working_sample.parquet")
    new = pd.read_parquet(P / "bigflow_working_sample_causal.parquet")
    assert old[["file_id", "file_row"]].equals(new[["file_id", "file_row"]])
    rows, look = [], []
    parts = []
    for path in list_files(bigflow_dir()):
        d = read_file(path, ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "Attack"])
        parts.append(pd.DataFrame({"source": d["source"], "src": d["ipv4_src_addr"].astype(str),
                                   "attack": d["attack"], "t": d["flow_start_milliseconds"].astype("int64")}))
    full = pd.concat(parts, ignore_index=True)
    for source in sorted(old["source"].unique()):
        m = (old["source"] == source).to_numpy()
        label = old.loc[m, "label"].to_numpy()
        rows += feature_rows("BigFlow", source, old.loc[m].reset_index(drop=True),
                             new.loc[m].reset_index(drop=True), label)
        s = old.loc[m, ["ipv4_src_addr", "attack", "label", "flow_start_milliseconds"]].rename(
            columns={"ipv4_src_addr": "src", "flow_start_milliseconds": "t"})
        s["src"] = s["src"].astype(str)
        la = lookahead(s, full[full["source"] == source])
        look.append(la.assign(dataset="BigFlow", source=source))
    return rows, look


def unsw():
    old = pd.read_parquet(P / "unsw_working_sample.parquet")
    new = pd.read_parquet(P / "unsw_working_sample_causal.parquet")
    rows = feature_rows("UNSW-NB15 (original)", "UNSW-NB15 (original)", old, new, old["label"].to_numpy())
    full = external.load_unsw()
    traffic = pd.DataFrame({"src": full["ipv4_src_addr"].astype(str), "attack": full["attack"],
                            "t": full["flow_start_milliseconds"].astype("int64")})
    s = old[["ipv4_src_addr", "attack", "label", "flow_start_milliseconds"]].rename(
        columns={"ipv4_src_addr": "src", "flow_start_milliseconds": "t"})
    s["src"] = s["src"].astype(str)
    s["t"] = s["t"].astype("int64")
    la = lookahead(s, traffic).assign(dataset="UNSW-NB15 (original)", source="UNSW-NB15 (original)")
    return rows, [la]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    r1, l1 = bigflow()
    r2, l2 = unsw()
    diff = pd.DataFrame(r1 + r2)
    diff.to_csv(REPORTS / "host_feature_diff.csv", index=False)

    look = pd.concat(l1 + l2, ignore_index=True)
    look_sum = look.groupby(["dataset", "source", "attack"]).agg(
        attack_flows=("future_same_episode", "size"),
        share_with_lookahead=("future_same_episode", lambda x: float((x > 0).mean())),
        median_future_flows=("future_same_episode", "median"),
        p95_future_flows=("future_same_episode", lambda x: float(np.quantile(x, 0.95))),
    ).reset_index()
    total = look.groupby(["dataset", "source"]).agg(
        attack_flows=("future_same_episode", "size"),
        share_with_lookahead=("future_same_episode", lambda x: float((x > 0).mean())),
        median_future_flows=("future_same_episode", "median"),
        p95_future_flows=("future_same_episode", lambda x: float(np.quantile(x, 0.95))),
    ).reset_index().assign(attack="(all attack flows)")
    look_sum = pd.concat([total, look_sum], ignore_index=True)
    look_sum.to_csv(REPORTS / "host_feature_lookahead.csv", index=False)

    # Summary
    lines = ["# Old (tumbling) vs new (past-only) host features\n",
             "Relative change = |new − old| / max(|old|, |new|). Full tables: `host_feature_diff.csv`, "
             "`host_feature_lookahead.csv`.\n", "## Per source and window (all flows)\n"]
    a = diff[diff.flows == "all"]
    w = a.groupby(["source", "window"]).agg(share_changed=("share_changed", "mean"),
                                           median_rel=("median_rel_change", "median"),
                                           p95_rel=("p95_rel_change", "median"),
                                           spearman=("spearman", "median")).reset_index()
    lines.append(md(w) + "\n")
    lines.append("Values are averages / medians over the 26 features of each window.\n")
    for group in ("benign", "attack"):
        lines.append(f"## 10 most-changed features, {group} flows (by mean relative change)\n")
        g = diff[diff.flows == group]
        for source, d in g.groupby("source"):
            top = d.sort_values("mean_rel_change", ascending=False).head(10)
            lines.append(f"**{source}:** " + ", ".join(
                f"`{f.replace('host_', '')}` ({v:.2f})" for f, v in zip(top.feature, top.mean_rel_change)) + "\n")
    lines.append("## Look-ahead of the old 60-second features into the flow's own attack episode\n")
    lines.append("Share of sampled attack flows whose old 60-s window contained later flows of the same "
                 "source host and same class (always in the same 1-minute-gap episode).\n")
    lines.append(md(total[["source", "attack_flows", "share_with_lookahead", "median_future_flows",
                           "p95_future_flows"]]) + "\n")
    (REPORTS / "host_feature_diff_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
