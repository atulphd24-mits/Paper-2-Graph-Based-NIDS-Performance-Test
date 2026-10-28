"""
Completeness check of Phase B (seeds 10-29, headline parts). Same checks as check_phase_a.py.
Usage: python scripts/check_phase_b.py   -> outputs/paper2_final/reports/phase_b_completeness.md
"""
import json

import pandas as pd

import check_phase_a as A

A.SEEDS = range(10, 30)
best = json.loads((A.OUT / "reports" / "endpoint_selection.json").read_text())["best_endpoint"]
A.PARTS = {
    "hl_ctx_main": ("context", "bigflow", A.BF, [A.EP1], [("HistGB", r) for r in A.REPS], ("model", "representation")),
    "hl_gated": A.PARTS["gated"],
}
for src in A.BF:
    tag = "hl_egs_ind_" + A.slug(src)
    A.PARTS[tag] = ("egraphsage", "bigflow", [src], [A.EP1],
                    [(A.BASE if hasattr(A, "BASE") else "HistGB (flow, no graph)", "-"),
                     (f"E-GraphSAGE ({best[A.slug(src)]})", "inductive")], ("model", "mode"))
    A.UNITS[tag] = 1
A.UNITS.update({"hl_ctx_main": 2, "hl_gated": 2})


def run_check(part):
    exp, dataset = A.PARTS[part][:2]
    meta = json.loads((A.OUT / "logs" / f"{exp}_{dataset}_final_{part}_meta.json").read_text())
    log = (A.OUT / "logs" / ("run_paper2_final_hl_graph.log" if "egs" in part else "run_paper2_final_hl_tabular.log")
           ).read_text(errors="replace")
    import re
    name = part if "egs" not in part else "hl_egs_ind"
    blocks = re.findall(rf"=== {name}[^\n]* start.*?=== {name}[^\n]* (?:end|FAILED)", log, re.S)
    src = A.PARTS[part][2][0] if "egs" in part else None
    block = next((b for b in blocks if src is None or src in b.splitlines()[0]), "")
    return {"part": part, "meta_status": meta["status"], "commit": meta["git_commit"][:7],
            "dirty": meta["git_dirty_tracked"], "seeds_10_29": meta["args"]["seeds"] == list(range(10, 30)),
            "log_end": "none" if not block else ("FAILED" if block.rstrip().endswith("FAILED") else "end"),
            "done_lines": len(re.findall(r"seed \d+ done$", block, re.M)), "expected_done": A.UNITS[part] * 20}


def main():
    datas = {"bigflow": A.Data("bigflow")}
    datas["bigflow"].subsample = False
    cache = {}
    runs = pd.DataFrame([run_check(p) for p in A.PARTS])
    res = pd.DataFrame([A.check_results(p) for p in A.PARTS])
    pred = pd.DataFrame([A.check_predictions(p, datas, cache) for p in A.PARTS])
    lines = ["# Phase B completeness check (seeds 10-29)\n", "Read-only; counts only, no detection metrics.\n",
             "## Run integrity\n", A.md(runs), "\n## Result rows\n", A.md(res), "\n## Prediction files\n", A.md(pred)]
    (A.OUT / "reports" / "phase_b_completeness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
