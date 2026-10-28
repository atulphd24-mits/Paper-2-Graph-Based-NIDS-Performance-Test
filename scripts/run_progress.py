"""
Progress of the full run against the measured one-seed times (runtime_measured.csv).

For each lane: finished parts (actual vs expected time) and the running part (units done, where a
unit is one "... seed N done" log line, i.e. one source x seed [x protocol]). Slowdown =
actual elapsed / expected elapsed for the work done so far.

Usage: python scripts/run_progress.py
"""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs/paper2_final/logs"
MEASURED = pd.read_csv(ROOT / "outputs/paper2_final/reports/runtime_measured.csv").set_index("part")["one_seed_min"]

# launch part -> (timing parts summed per seed, units per seed)
PARTS = {
    "splits": (["splits"], 2), "ctx_main": (["ctx_hgb_main", "ctx_rfmlp_main"], 2),
    "ctx_gap5": (["ctx_hgb_gap5"], 2), "ctx_chrono": (["ctx_hgb_chrono"], 2), "ctx_e0": (["ctx_hgb_e0"], 2),
    "gated": (["gated"], 2), "unsw_splits": (["unsw_splits"], 1), "unsw_context": (["unsw_context"], 1),
    "ciciot_splits": (["ciciot_splits"], 1), "ciciot_context": (["ciciot_context"], 1),
    "egs_ind": (["egs_ind_gap1", "egs_ind_random"], 4), "egs_tr": (["egs_tr_gap1", "egs_tr_random"], 4),
    "unsw_egs": (["unsw_egs"], 2),
}
SEEDS = 10
STAMP = re.compile(r"^=== (\S+) (start|end|FAILED) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def lane(path, now):
    rows, current, units = [], None, 0
    for line in path.read_text(errors="replace").splitlines():
        m = STAMP.match(line)
        if m:
            part, what, ts = m[1], m[2], datetime.strptime(m[3], "%Y-%m-%d %H:%M:%S")
            if what == "start":
                current, start, units = part, ts, 0
            else:
                rows.append((part, what, (ts - start).total_seconds() / 3600, 1.0))
                current = None
        elif current and re.search(r"seed \d+ done$", line):
            units += 1
    if current:
        per_seed = PARTS[current][1]
        rows.append((current, "running", (now - start).total_seconds() / 3600, units / (per_seed * SEEDS)))
    out = []
    for part, status, hours, frac in rows:
        names, _ = PARTS[part]
        expected = sum(MEASURED[n] for n in names) * SEEDS / 60
        out.append({"part": part, "status": status, "elapsed_h": hours, "fraction_done": frac,
                    "expected_h_total": expected, "expected_h_so_far": expected * frac})
    return pd.DataFrame(out)


def main():
    now = datetime.now()
    for name in ("tabular", "graph"):
        t = lane(LOGS / f"run_paper2_final_{name}.log", now)
        done = t[t.fraction_done > 0]
        slow = done.elapsed_h.sum() / done.expected_h_so_far.sum() if len(done) else float("nan")
        print(f"\n{name} lane: slowdown {slow:.2f}x (actual / expected for work done)")
        print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
