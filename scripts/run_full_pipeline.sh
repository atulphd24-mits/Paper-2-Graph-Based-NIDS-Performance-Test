#!/usr/bin/env bash
# Re-runs every experiment of the leakage study, in dependency order.
# Usage: nohup bash scripts/run_full_pipeline.sh > outputs/reports/leakage_study/run_full_pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/reports/leakage_study
TEN="0 1 2 3 4 5 6 7 8 9"

step() { echo "=== $(date '+%F %T') $1"; }

step "label ambiguity (all flows)";      $PY scripts/label_ambiguity.py            > $LOG/run_label_ambiguity.log 2>&1
step "working sample + host features";   $PY scripts/build_working_sample.py       > $LOG/run_build_sample.log 2>&1
step "part 1 shortcuts";                 $PY scripts/leakage_study.py --parts 1 --seeds 0 1 2 > $LOG/run_part1.log 2>&1
step "parts 3-4 splits + context";       $PY scripts/leakage_study.py --parts 3 4 --seeds 0 1 2 3 4 --part4-seeds $TEN > $LOG/run_part34.log 2>&1
step "part 5 two-branch";                $PY scripts/leakage_study.py --parts 5 --part4-seeds $TEN > $LOG/run_part5.log 2>&1
step "robustness A (RF, MLP)";           $PY scripts/robustness_study.py --parts A > $LOG/run_robustA.log 2>&1
step "robustness B (gap)";               $PY scripts/robustness_study.py --parts B --gap-seeds $TEN > $LOG/run_robustB.log 2>&1
step "E-GraphSAGE";                      $PY scripts/egraphsage_study.py --seeds $TEN --epochs 300 --threads 8 > $LOG/run_egraphsage.log 2>&1
step "figures and tables";               $PY scripts/make_paper_figures.py > $LOG/run_figures.log 2>&1
step "paper LaTeX tables + figures";     $PY scripts/make_paper_tables.py > $LOG/run_paper_tables.log 2>&1
step "done"
