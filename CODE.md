# Paper 2 code

Copied from the SOC-Threat-Detection repository. Run everything from this folder (scripts resolve paths relative
to it); the raw data (`data/`) is not included.

| Folder / file | Contents |
|---|---|
| `PROTOCOL_FREEZE.md` | frozen protocol of the final run and its corrections |
| `requirements.txt`, `configs/config.yaml` | packages and dataset/feature configuration |
| `src/` | library code: `data` (BigFlow loading, stratified sample), `features` (causal host features), `graph` (edges, neighbourhood and host features), `models` (evaluation, predictions, two-branch, E-GraphSAGE), `stats` (Holm, bootstrap, Wilcoxon), `utils` |
| `tests/` | unit tests for the statistics and test subsets: `python -m unittest discover -s tests` |

## `scripts/` (kept in one folder: several scripts import each other)

**Final run (paper as submitted)**

| Step | Scripts |
|---|---|
| Data preparation | `build_stratified_sample.py`, `build_episode_bounds.py` |
| Checks and smoke tests | `check_phase_a.py`, `check_phase_b.py`, `smoke_compare.py`, `smoke_compare_strat.py`, `host_feature_diff.py`, `prediction_disk_estimate.py`, `select_endpoint.py` |
| Main run | `run_paper2_final.sh` → `paper2_final.py` (splits, context features, gated model, E-GraphSAGE, external sources) |
| Statistics and post-hoc | `paper2_stats.py`, `per_family_tpr.py`, `tie_sensitivity.py`, `posthoc_split_decomposition.py`, `posthoc_ambiguity_removed.py` |
| Paper outputs | `make_paper2_tables.py`, `make_paper2_figures.py`, `make_export.py` |
| Run monitoring | `run_progress.py`, `timing_table.py`, `time_one_seed.sh` |

**Earlier version (v1 baseline, `outputs/paper2_final/v1_baseline/`)**

`run_full_pipeline.sh` → `build_working_sample.py`, `build_causal_sample.py`, `label_ambiguity.py`,
`leakage_study.py`, `robustness_study.py`, `egraphsage_study.py`, `external_study.py`, `make_paper_tables.py`,
`make_paper_figures.py`.

`make_paper2_tables.py`, `make_paper2_figures.py`, `check_phase_b.py` and `run_progress.py` read result files
(`outputs/paper2_final/{results,reports,stats}`) that this copy does not contain.
