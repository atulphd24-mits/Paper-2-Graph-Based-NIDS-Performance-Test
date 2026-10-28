"""
Measured runtime tables for options (b-i) and (b-ii), from the one-seed timing runs
(scripts/time_one_seed.sh). Each part's wall time is read from its *_meta.json and scaled by the
number of seeds in the frozen run plan (PROTOCOL_FREEZE.md, section 8):

    10 seeds for every part; 20 further seeds (30 in total) for the headline parts at the main
    protocol (context HistGB, gated detector, inductive E-GraphSAGE at the main gap).
    (b-i) includes inductive E-GraphSAGE at the 5-min gap; (b-ii) does not.

Writes outputs/paper2_final/reports/runtime_measured.{csv,md}.
"""
import json
from pathlib import Path

import pandas as pd

from smoke_compare import md

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs/paper2_final/logs"
R = ROOT / "outputs/paper2_final/reports"

# part -> (experiment_dataset prefix, description, lane, headline, in b-i, in b-ii)
PARTS = {
    "splits": ("splits_bigflow", "Split inflation: random / episodes / chronological, HistGB+RF+MLP + timestamp-only", "tabular", False, True, True),
    "ctx_hgb_main": ("context_bigflow", "Context features, HistGB, main protocol (G=1, E=5)", "tabular", True, True, True),
    "ctx_rfmlp_main": ("context_bigflow", "Context features, RF + MLP, main protocol", "tabular", False, True, True),
    "ctx_hgb_gap5": ("context_bigflow", "Context features, HistGB, G=5, E=5", "tabular", False, True, True),
    "ctx_hgb_chrono": ("context_bigflow", "Context features, HistGB, chronological reference", "tabular", False, True, True),
    "ctx_hgb_e0": ("context_bigflow", "Context features, HistGB, G=1, E=0", "tabular", False, True, True),
    "gated": ("gated_bigflow", "Family-gated detector (reused + split validation)", "tabular", True, True, True),
    "egs_ind_gap1": ("egraphsage_bigflow", "E-GraphSAGE inductive, main protocol, 3 endpoints + HistGB baseline", "graph", True, True, True),
    "egs_ind_random": ("egraphsage_bigflow", "E-GraphSAGE inductive, random rows", "graph", False, True, True),
    "egs_ind_gap5": ("egraphsage_bigflow", "E-GraphSAGE inductive, G=5, E=5", "graph", False, True, False),
    "egs_tr_gap1": ("egraphsage_bigflow", "E-GraphSAGE transductive, main protocol", "graph", False, True, True),
    "egs_tr_random": ("egraphsage_bigflow", "E-GraphSAGE transductive, random rows", "graph", False, True, True),
    "unsw_splits": ("splits_unsw", "UNSW-NB15: split inflation", "tabular", False, True, True),
    "unsw_context": ("context_unsw", "UNSW-NB15: context features, HistGB+RF+MLP, main protocol", "tabular", False, True, True),
    "unsw_egs": ("egraphsage_unsw", "UNSW-NB15: E-GraphSAGE inductive + transductive, random + main", "graph", False, True, True),
    "ciciot_splits": ("splits_ciciot", "CIC-IoT2023: random split", "tabular", False, True, True),
    "ciciot_context": ("context_ciciot", "CIC-IoT2023: flow + neighbourhood, random split", "tabular", False, True, True),
}
SEEDS, HEADLINE_EXTRA = 10, 20


def main():
    rows = []
    for part, (prefix, desc, lane, headline, b1, b2) in PARTS.items():
        f = LOGS / f"{prefix}_timing_{part}_meta.json"
        meta = json.loads(f.read_text()) if f.exists() else {}
        sec = meta.get("wall_clock_s") if meta.get("status") == "finished" else None
        seeds = SEEDS + (HEADLINE_EXTRA if headline else 0)
        rows.append({"part": part, "description": desc, "lane": lane, "status": meta.get("status", "missing"),
                     "one_seed_min": None if sec is None else sec / 60, "seeds": seeds,
                     "total_h": None if sec is None else sec * seeds / 3600, "in_b_i": b1, "in_b_ii": b2})
    t = pd.DataFrame(rows)
    t.to_csv(R / "runtime_measured.csv", index=False)

    out = []
    for opt, col in (("(b-i)", "in_b_i"), ("(b-ii)", "in_b_ii")):
        sub = t[t[col]]
        lanes = sub.groupby("lane")["total_h"].sum()
        out.append({"option": opt, "sequential_h": sub["total_h"].sum(),
                    "tabular_lane_h": lanes.get("tabular", 0.0), "graph_lane_h": lanes.get("graph", 0.0),
                    "two_lanes_h (max)": lanes.max()})
    opts = pd.DataFrame(out)
    opts.to_csv(R / "runtime_measured_options.csv", index=False)
    lines = ["# Measured runtimes (one seed, sequential, full working sample)\n",
             "Wall-clock time of seed 0 for every part, scaled by the seeds in the frozen plan "
             f"({SEEDS} seeds; {SEEDS + HEADLINE_EXTRA} for headline parts). Machine: Xeon Silver 4310, "
             "24 cores, no GPU.\n",
             md(t.drop(columns=["lane"]), 2), "\n## Options\n",
             "`two_lanes_h`: tabular and graph runs in two parallel lanes; the lane totals are measured "
             "sequentially, so contention between lanes is not included.\n", md(opts, 1)]
    (R / "runtime_measured.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
