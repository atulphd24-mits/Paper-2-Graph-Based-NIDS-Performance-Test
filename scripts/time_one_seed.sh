#!/bin/sh
# One seed (seed 0) of every experiment in the frozen run plan, run sequentially so that
# wall-clock times are not affected by other jobs. Each run writes
# outputs/paper2_final/logs/<experiment>_<dataset>_timing_<part>_meta.json with its wall time;
# scripts/timing_table.py turns these into the (b-i)/(b-ii) runtime tables.
# Results go to outputs/paper2_final/results/*_timing_*.csv and are not used in the paper.
#
# Usage: nohup sh scripts/time_one_seed.sh > outputs/paper2_final/logs/time_one_seed.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
D="$PY scripts/paper2_final.py"
S="--seeds 0"

run() {
    part=$1; shift
    echo "=== $part start $(date +%T)"
    "$@" || echo "=== $part FAILED"
    echo "=== $part end $(date +%T)"
}

# BigFlow (ToN-IoT, CSE-CIC-IDS2018)
run splits         $D splits     --dataset bigflow $S --models HistGB RandomForest MLP --tag _timing_splits
run ctx_hgb_main   $D context    --dataset bigflow $S --models HistGB --tag _timing_ctx_hgb_main
run ctx_rfmlp_main $D context    --dataset bigflow $S --models RandomForest MLP --tag _timing_ctx_rfmlp_main
run ctx_hgb_gap5   $D context    --dataset bigflow $S --models HistGB --gap 5 --tag _timing_ctx_hgb_gap5
run ctx_hgb_chrono $D context    --dataset bigflow $S --models HistGB --protocol chronological --tag _timing_ctx_hgb_chrono
run ctx_hgb_e0     $D context    --dataset bigflow $S --models HistGB --embargo 0 --tag _timing_ctx_hgb_e0
run gated          $D gated      --dataset bigflow $S --tag _timing_gated
run egs_ind_gap1   $D egraphsage --dataset bigflow $S --modes inductive --split-protocols episodes --with-baseline --tag _timing_egs_ind_gap1
run egs_ind_random $D egraphsage --dataset bigflow $S --modes inductive --split-protocols random --with-baseline --tag _timing_egs_ind_random
run egs_ind_gap5   $D egraphsage --dataset bigflow $S --modes inductive --split-protocols episodes --gap 5 --with-baseline --tag _timing_egs_ind_gap5
run egs_tr_gap1    $D egraphsage --dataset bigflow $S --modes transductive --split-protocols episodes --tag _timing_egs_tr_gap1
run egs_tr_random  $D egraphsage --dataset bigflow $S --modes transductive --split-protocols random --tag _timing_egs_tr_random

# External: UNSW-NB15 (main gap only), CIC-IoT2023 (random split only, no timestamps)
run unsw_splits    $D splits     --dataset unsw $S --models HistGB RandomForest MLP --tag _timing_unsw_splits
run unsw_context   $D context    --dataset unsw $S --models HistGB RandomForest MLP --tag _timing_unsw_context
run unsw_egs       $D egraphsage --dataset unsw $S --modes inductive transductive --tag _timing_unsw_egs
run ciciot_splits  $D splits     --dataset ciciot $S --models HistGB RandomForest MLP --tag _timing_ciciot_splits
run ciciot_context $D context    --dataset ciciot $S --models HistGB RandomForest MLP --protocol random --tag _timing_ciciot_context
echo FINISHED
