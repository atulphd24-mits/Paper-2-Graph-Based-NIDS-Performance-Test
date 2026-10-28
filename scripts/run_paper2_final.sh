#!/bin/sh
# Full paper-2 run, option (b-ii), frozen protocol (PROTOCOL_FREEZE.md).
#
# Phase A (seeds 0-9) in two parallel lanes:
#   tabular lane: splits, context (main, gap 5, chronological, E=0), gated, external tabular
#   graph lane:   E-GraphSAGE inductive + transductive (BigFlow), UNSW-NB15 E-GraphSAGE
# Phase B (seeds 10-29, after both lanes finish), headline comparisons at the main protocol:
#   context HistGB, gated detector, inductive E-GraphSAGE with the endpoint chosen per source on
#   VALIDATION PR-AUC over seeds 0-9 (scripts/select_endpoint.py) + flow-only HistGB baseline.
#
# Every part writes metrics to outputs/paper2_final/results/*_final_*.csv, per-flow predictions to
# outputs/paper2_final/predictions/, and a *_meta.json with the commit it started from.
#
# Usage: nohup sh scripts/run_paper2_final.sh > outputs/paper2_final/logs/run_paper2_final.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
D="$PY scripts/paper2_final.py"
L=outputs/paper2_final/logs
A="--seeds 0-9"
B="--seeds 10-29"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "tracked files modified: commit before launching"; exit 1
fi
echo "commit $(git rev-parse HEAD) start $(date '+%F %T')"

run() {
    part=$1; shift
    echo "=== $part start $(date '+%F %T')"
    if "$@"; then echo "=== $part end $(date '+%F %T')"; else echo "=== $part FAILED $(date '+%F %T')"; fi
}

tabular_lane() {
    run splits          $D splits     --dataset bigflow $A --models HistGB RandomForest MLP --tag _final_splits
    run ctx_main        $D context    --dataset bigflow $A --models HistGB RandomForest MLP --tag _final_ctx_main
    run ctx_gap5        $D context    --dataset bigflow $A --models HistGB --gap 5 --tag _final_ctx_gap5
    run ctx_chrono      $D context    --dataset bigflow $A --models HistGB --protocol chronological --tag _final_ctx_chrono
    run ctx_e0          $D context    --dataset bigflow $A --models HistGB --embargo 0 --tag _final_ctx_e0
    run gated           $D gated      --dataset bigflow $A --tag _final_gated
    run unsw_splits     $D splits     --dataset unsw $A --models HistGB RandomForest MLP --tag _final_unsw_splits
    run unsw_context    $D context    --dataset unsw $A --models HistGB RandomForest MLP --tag _final_unsw_context
    run ciciot_splits   $D splits     --dataset ciciot $A --models HistGB RandomForest MLP --tag _final_ciciot_splits
    run ciciot_context  $D context    --dataset ciciot $A --models HistGB RandomForest MLP --protocol random --tag _final_ciciot_context
    echo "=== tabular lane done $(date '+%F %T')"
}

graph_lane() {
    run egs_ind         $D egraphsage --dataset bigflow $A --modes inductive --split-protocols episodes random --with-baseline --tag _final_egs_ind
    run egs_tr          $D egraphsage --dataset bigflow $A --modes transductive --split-protocols episodes random --tag _final_egs_tr
    run unsw_egs        $D egraphsage --dataset unsw $A --modes inductive transductive --split-protocols episodes random --tag _final_unsw_egs
    echo "=== graph lane done $(date '+%F %T')"
}

tabular_lane > $L/run_paper2_final_tabular.log 2>&1 &
TAB=$!
graph_lane > $L/run_paper2_final_graph.log 2>&1 &
GRAPH=$!
wait $TAB
wait $GRAPH
grep -h "===" $L/run_paper2_final_tabular.log $L/run_paper2_final_graph.log
echo "phase A done $(date '+%F %T')"

# ---------------- Phase B: headline seeds 10-29 ----------------
run select_endpoint $PY scripts/select_endpoint.py --tag _final_egs_ind --seeds 0-9 || exit 1
headline_tabular() {
    run hl_ctx_main     $D context    --dataset bigflow $B --models HistGB --tag _final_hl_ctx_main
    run hl_gated        $D gated      --dataset bigflow $B --tag _final_hl_gated
}
headline_graph() {
    for source in ToN-IoT CSE-CIC-IDS2018; do
        endpoint=$($PY -c "import json,sys; from src.models.predictions import slug; \
print(json.load(open('outputs/paper2_final/reports/endpoint_selection.json'))['best_endpoint'][slug('$source')])")
        run "hl_egs_ind $source $endpoint" $D egraphsage --dataset bigflow $B --sources "$source" --modes inductive \
            --endpoints "$endpoint" --split-protocols episodes --with-baseline --tag "_final_hl_egs_ind_$(echo $source | tr -c 'A-Za-z0-9\n' '_')"
    done
}
headline_tabular > $L/run_paper2_final_hl_tabular.log 2>&1 &
TAB=$!
headline_graph > $L/run_paper2_final_hl_graph.log 2>&1 &
GRAPH=$!
wait $TAB
wait $GRAPH
grep -h "===" $L/run_paper2_final_hl_tabular.log $L/run_paper2_final_hl_graph.log
echo "FINISHED $(date '+%F %T')"
