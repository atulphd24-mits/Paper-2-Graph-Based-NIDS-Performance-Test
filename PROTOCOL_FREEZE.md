# Protocol freeze — paper 2 final

Frozen on 2026-10-14 in the commit that adds this file (branch `paper2-final`).
**No protocol changes after this commit.** (Later commits may only clarify the wording so it
matches the frozen code; each such commit says so.) Any later deviation must be reported as a
deviation in the paper, with its reason, and must not overwrite the frozen-protocol results.

Earlier results (before the leakage fixes) are tagged `v1-pre-fix`; their CSVs are kept in
`outputs/paper2_final/v1_baseline/`.

## 1. Data

| Dataset | Use |
|---|---|
| BigFlow-NIDS v2 (DOI 10.17632/nv729tbdgz.2), 67 files, 66,935,021 flows | main; sources ToN-IoT and CSE-CIC-IDS2018 |
| BigFlow's BoT-IoT and UNSW-NB15 parts | stated limitations only (99.7% attacks; saturated) |
| UNSW-NB15 original release (2,540,047 records) | external validation, full protocol, main gap only |
| CIC-IoT2023 feature CSVs (46,686,579 rows) | external validation, partial protocol (no timestamps or addresses: random split, no host features, no E-GraphSAGE) |

Source of each BigFlow flow: timestamp period (UNSW-NB15 < 2016; CSE-CIC-IDS2018 Jan–14 Apr 2018;
BoT-IoT 15 Apr–Dec 2018; ToN-IoT 2019).

## 2. Working sample (`scripts/build_stratified_sample.py`)

- **Attack classes:** per (source, class), cap 20,000 flows, **episode-stratified**: keep
  min(20, episode size) flows from every full-data episode (1-min gap), then fill the rest of the
  quota at random. Classes with ≤ 20,000 flows are kept entirely.
- **Benign:** the v1 working-sample rows (about 100,000 per source).
- **Design weights:** every row carries `design_weight` = 1 / inclusion probability (constant within
  an episode for attacks, within a source for benign). **All prevalence-weighted metrics use these
  weights**, so they estimate full-data rates. The weights sum to the full-data flow counts
  (checked per source and class).
- **Files** (`data/processed/`, built by `scripts/build_stratified_sample.py`, the driver's
  defaults): `bigflow_working_sample_strat_causal.parquet` (860,742 rows, all four BigFlow sources)
  and `unsw_working_sample_strat_causal.parquet` (196,605 rows). CIC-IoT2023 has no timestamps, so
  it has no episodes and is **not** stratified: `ciciot_working_sample.parquet` (v1 random per-class
  cap) with class-level prevalence weights.
- **Full working sample** per source is used for every experiment (no 60k subsample).
- **Sampling columns are never model inputs:** `design_weight`, `episode_full`, `file_id`,
  `file_row` (`SAMPLING_COLUMNS` in `scripts/paper2_final.py`; tested in
  `tests/test_model_features.py`). Every run records the sample file, row count, whether it is
  episode-stratified and the feature counts in its `*_meta.json` (`sample_info`).
- **Effect on subset (i)** (smoke check, main protocol, seeds 0-4, mean; full table in
  `outputs/paper2_final/reports/smoke_strat_vs_full.md`):

  | Source | Sample | Test episodes in (i) | Families in (i) | Attack flows in (i) | Share of test attacks |
  |---|---|---|---|---|---|
  | ToN-IoT | 20k cap (previous) | 18.6 (min 14) | 4.2 (3-5) | 6,533 | 0.33 |
  | ToN-IoT | episode-stratified | 36.2 (min 31) | 4.2 (3-5) | 6,929 | 0.35 |
  | CSE-CIC-IDS2018 | 20k cap (previous) | 38.8 (min 35) | 3.0 (3-3) | 1,042 | 0.04 |
  | CSE-CIC-IDS2018 | episode-stratified | 28.6 (min 25) | 2.8 (2-3) | 4,099 | 0.13 |
  | UNSW-NB15 (original) | 20k cap (previous) | 89.4 (min 76) | 5.4 (5-6) | 10,527 | 0.68 |
  | UNSW-NB15 (original) | episode-stratified | 82.0 (min 65) | 5.0 (4-6) | 9,468 | 0.59 |

  ToN-IoT scanning now keeps all 173 full-data episodes (15 before). CSE-CIC-IDS2018 Infiltration
  has fewer test episodes (3.2 vs 13) but six to seven times more test flows (3,900-4,900 vs 638): under the 20k cap no
  constrained draw passed and largest-first kept the three large episodes in training, leaving
  only 13 tiny episodes (638 flows) for test; with stratification the constrained draws pass and
  a large episode is held out. UNSW-NB15 differences are seed noise (families at the 500-flow and
  3-episode limits), since its large classes have few episodes.

## 3. Features

- **Behaviour features:** all columns except flow start/end timestamps, source/destination IP,
  source port, DNS query id and the sampling columns (BigFlow: 47 features; UNSW-NB15: 41;
  CIC-IoT2023: 46).
- **Host features (past-only), 78:** for a flow at time t, statistics of its source host,
  destination host and source–destination pair over flows that start in (t − w, t] for
  w = 1 s, 10 s, 60 s, built over **all flows of the source (every split)**; no labels used. The flow
  itself is counted; flows with the same timestamp are counted only if they come earlier in the
  stream order (file, then row). Future flows never change a feature.
- **Neighbourhood features (inductive):** k = 20. Scaler, k-NN index, distance features and LOF are
  fitted on **training rows only**; validation and test rows are queried against the training
  index. Label features use training labels only, leave-one-out for training rows.

## 4. Splits (`src/models/evaluation.py`)

70% train / 15% validation / 15% test targets, per source dataset.

- **Episodes:** from **full-data** timestamps of each (source, class); a new episode starts after a
  gap of more than G. Sampled flows are mapped onto these episodes.
- **Main protocol: G = 1 min, embargo E = 5 min.**
- **Episode assignment (per seed): constrained random.** Each episode goes independently to
  train/validation/test with probabilities 0.70/0.15/0.15; a draw is accepted if every split has
  ≥ 1 episode, training holds ≥ 50% of the class's flows, and validation and test each hold 10–25%.
  Up to 1,000 draws; otherwise **largest-first fallback** (largest episode to training, then each
  episode to the split furthest below target, ≥ 1 episode per split), logged per class and marked
  "dominant episode always in training".
- **Chronological fallback:** classes with < 3 full-data episodes are split by time within the
  class (earliest 70% train, latest 15% test).
- **Benign:** 30-minute blocks assigned like episodes.
- **Embargo (purge):**
  - test flows are never purged;
  - a training flow of class c is purged if it lies within **5 min** of a validation or test unit
    of class c;
  - a validation flow of class c is purged if it lies within **5 min** of a test unit of class c,
    except for chronological-fallback classes, where this validation/test buffer is **1 min**
    (the longest host-feature window);
  - benign blocks follow the same rule among themselves;
  - **benign host rule:** a benign training flow is purged if its source or destination IP occurs
    (as source or destination) in a held-out attack flow of the same source dataset that started
    at most 60 s before it (checked on the sampled flows).
- **Sensitivity / reference settings:** G = 5 min with E = 5 min; **chronological split**
  (every class split by time, E = 5 min) as a reference protocol, not an episode setting;
  E = 0 (no embargo) at G = 1 min for HistGB only.
- **Random rows** (stratified by class) for the split-inflation comparison.

## 5. Test subsets and thresholds

Every result is evaluated on three test subsets with the **same** alert threshold:

| Subset | Attack flows | Benign flows |
|---|---|---|
| **(i) cross-episode** (primary) | from real episodes of classes with **≥ 3 test episodes** and **≥ 500 training flows** | all test benign |
| (ii) fallback | from chronological-fallback classes | all test benign |
| (iii) all | all test attack flows | all test benign |

Classes excluded from (i) are listed per seed and reported as within-episode only.
The FPR reached on test is reported for every subset. Significance tests use (i) and (iii).

- **Alert thresholds:** pooled, the (1 − FPR) quantile of **validation** benign scores, applied
  unchanged to test (FPR = 1% primary, 0.1% secondary). No per-class thresholds.
- **Family-gated detector:** per-family expert choice on validation data; a family with
  **< 50 validation flows** uses the pooled default expert (flow features). Two versions:
  *split validation* (primary: one half of validation, disjoint by episode, chooses experts, the
  other half sets thresholds) and *reused validation* (reported for comparison).

## 6. Metrics and statistics

- **Primary metric: design-weighted TPR at 1% FPR on subset (i).**
- Secondary: TPR at 0.1% FPR, design-weighted PR-AUC and F1 (F1 threshold tuned on weighted
  validation rows), ROC-AUC, FPR reached on test.
- **Episode-level cluster bootstrap** (within each seed) for subset (i) TPR: resample test episodes
  with replacement (2,000 resamples), 95% percentile CI, n = number of test episodes.
- **Across seeds:** paired two-sided Wilcoxon signed-rank tests; **Holm–Bonferroni** within declared
  families (`outputs/paper2_final/reports/stats_protocol.md`); paired bootstrap 95% CI of mean
  differences over seeds (10,000 resamples).
- **Seeds:** 10 for every comparison; 30 for the headline comparisons at the main protocol
  (flow vs flow + host vs flow + neighbourhood + host for HistGB; gated vs flow vs best single
  expert vs unweighted ensemble; inductive E-GraphSAGE (best endpoint) vs flow-only HistGB),
  run after the 10-seed results.
- **Best E-GraphSAGE endpoint** (for the 30-seed runs): chosen per source on **validation rows
  only**: the real endpoint (IP or IP:port; the random control is never a candidate) with the
  highest mean design-weighted validation PR-AUC of the inductive model at the main protocol over
  seeds 0-9 (`scripts/select_endpoint.py`, from the saved predictions; test rows are not read).

### Secondary response-time metrics (research objective 2)

**Scope (decision of 2026-10-16, reporting only):** paper 2 covers research objective 1 only.
Metrics (a)–(d) are **not reported in paper 2** (no table, figure or abstract claim). After the
run, (a)–(c) are computed from the saved predictions into `outputs/paper3_baseline/` as the
baseline for paper 3; (d) latency is left to paper 3. Paper 2's scope note states that these
metrics were pre-specified in this file and are reported in a companion paper (paper 3).

Added before launch and before any result was seen. (The one-seed timing runs wrote detection
metrics to `outputs/paper2_final/results/*_timing_*.csv`; none of them has been inspected, and no
response-time metric has been computed.) They do not change the primary metric or any
selection rule. Metrics (a)–(c) are computed **after the run** from the saved per-flow
predictions (section 8); (d) is measured separately.

Common definitions:
- **Threshold τ:** the same alert threshold as the primary metric, the pooled 99th percentile of the
  run's validation benign scores (FPR 1%); for the gated detector with split validation, the
  threshold half. **Alert:** a test flow with score > τ (the rule used for TPR).
- **Alert time:** the end time of the alerted flow (`flow_end_milliseconds`), the earliest moment a
  flow-based detector can score it. Processing time is excluded here and covered by (d).
- **Scope:** runs with timestamps and episodes: the main protocol and the G = 5 and E = 0 HistGB
  runs. Not defined for random rows, the chronological reference or CIC-IoT2023 (no timestamps).
- **Sampling caveat:** only sampled flows are scored, so a sampled episode can only alert on its
  sampled flows. (a) is therefore an upper bound on the full-data time to detect, and (b) a
  lower bound on the full-data detection rate. Both are exact for episodes kept entirely
  (≤ 20 flows, or classes ≤ 20,000 flows).

(a) **Per-episode time to detect (TTD):** for each test episode in subset (i), TTD = alert time of
its first alerted flow − the episode's first flow in the full data (`episode_start_ms`, from
the full-data episode bounds). An episode with no alert has TTD = ∞ (censored, kept in the
median). Reported per family as the **median TTD** over all (seed, episode) test units pooled
over seeds, with a 95% **episode cluster bootstrap CI** (2,000 resamples of distinct episode
ids; an episode that is in test under several seeds is one cluster). If more than half of the
units are undetected, the median is reported as "not detected". Fallback classes (subset (ii))
are reported separately and labelled within-episode, with start = the first sampled test flow
of the class in that seed.

(b) **Episode detection rate:** share of subset (i) test episodes with at least one alert,
unweighted over episodes, per family and overall, pooled over seeds with the same episode
cluster bootstrap CI.

(c) **Benign alerts per hour:** per source and seed, the design-weighted count of test benign
flows with score > τ (an estimate of full-data alerts) divided by the test benign time,
0.5 h × the number of test 30-minute benign blocks that contain at least one sampled benign
flow. Reported as the mean over seeds with a bootstrap CI over seeds (10,000 resamples).

(d) **Per-flow inference latency per model:** measured separately, after the run, on seed-0
models refitted by the latency script with the same code and seed (not used for any detection
metric), on this machine (Xeon Silver 4310, 24 cores, no GPU) with nothing else running.
Reported in µs per flow, median of 5 repeats, at 1 thread and 24 threads:
- model scoring on precomputed features, for the whole test set in one batch and for 1,000
  single-flow calls;
- feature cost per flow: past-only host features (streaming update), neighbourhood features
  (k-NN query against the training index), and for E-GraphSAGE the forward pass over the
  inference graph divided by the number of test edges.

## 7. Models (fixed hyper-parameters, never tuned on test)

- HistGradientBoosting: up to 300 iterations, learning rate 0.1, balanced class weights,
  scikit-learn's automatic early stopping (internal 10% of training rows when > 10,000 rows).
- Random forest: 200 trees, min_samples_leaf 2, sqrt features, balanced_subsample.
- MLP: signed-log transform + standard scaling (fitted on training rows), (128, 64), Adam 1e-3,
  alpha 1e-4, batch 512, ≤ 100 epochs, early stopping on an internal 10% of training rows
  (8 epochs without improvement), minority class oversampled with replacement to balance.
- E-GraphSAGE (re-implemented): 2 layers, hidden 128, dropout 0.2, Adam 1e-3, weight decay 1e-5,
  ≤ 300 epochs, early stopping on validation PR-AUC (patience 30); endpoints IP, IP:port, random
  control; **inductive** (training graph = training edges only) and **transductive** modes.
- Family-gated detector: experts = HistGB on flow, flow + neighbourhood, flow + host; router =
  multiclass HistGB (200 iterations, no early stopping).

### Where design weights are and are not used (as implemented at the freeze; not changed)

- **Training: design weights are not used.** Every model is fitted on unweighted training rows;
  class imbalance is handled only by the class balancing listed above (HistGB `balanced`, random
  forest `balanced_subsample`, MLP minority oversampling, E-GraphSAGE inverse-frequency class
  weights in the cross-entropy loss). Training therefore sees small episodes over-represented
  relative to the full data, as the stratified sample intends.
- **Family-gated detector:** the per-family expert choice compares detection rates on
  **unweighted** validation flows of each family; the router is trained unweighted.
- **Alert thresholds:** (1 − FPR) quantile of **unweighted** validation benign scores. Benign
  design weights are constant within a source and evaluation is per source, so this equals the
  weighted quantile.
- **Weighted (design weights):** F1 threshold tuning on validation rows, and every reported
  test metric (TPR, PR-AUC, F1, precision) including the episode-level bootstrap.

## 8. Run plan

Main protocol (G = 1, E = 5): splits (HistGB, RF, MLP + timestamp-only), context (HistGB, RF, MLP),
gated detector, E-GraphSAGE inductive and transductive, external datasets.
HistGB only: G = 5 / E = 5, chronological reference, G = 1 / E = 0.
E-GraphSAGE inductive also at G = 5 / E = 5 in option (b-i).

**Chosen: option (b-ii)** (E-GraphSAGE at the main gap only), launched with
`scripts/run_paper2_final.sh` from a clean tree: phase A (seeds 0-9) in two parallel lanes
(tabular; graph), then phase B (seeds 10-29) for the headline comparisons, with the E-GraphSAGE
endpoint chosen as in section 6.

Driver: `scripts/paper2_final.py`. Every run writes per-seed rows to
`outputs/paper2_final/results/` and a `*_meta.json` with package versions, git commit and
wall-clock time. Every run also saves **per-flow validation and test predictions** for every seed
(`src/models/predictions.py`): one zstd-compressed parquet file per (experiment, dataset, source,
seed, protocol[, gated validation version]) under `outputs/paper2_final/predictions/`, with row
index, split, flow start and end time, episode id, full-data episode start, split mode, family,
true label, design weight, subset (i)/(ii) membership and one float32 score column per
model/representation. Files are written once and never overwritten, and are **kept permanently**:
paper 3 depends on them.

## 9. Post-freeze corrections

Each correction enforces what this document already states; none changes a rule, setting,
threshold or metric. The one addition (secondary response-time metrics) is marked as such and
leaves every primary rule unchanged.

| Commit | Correction |
|---|---|
| 5a6dea8 | Wording: HistGB and MLP early stopping described as implemented. |
| 1bd46de | **Bug fix (data leak), made before any results were seen:** on the stratified BigFlow sample the driver used `design_weight` and `episode_full` as features (49 instead of the frozen 47); both identify the episode. Found while checking the one-seed timing run, which was stopped before any experiment finished; no metric from it was looked at or used. Its partial outputs are kept in `outputs/paper2_final/invalid_leaky_features/`. Fix: `SAMPLING_COLUMNS` excluded for every dataset, test on the frozen feature counts. Also: sample files and stratification scope listed explicitly; smoke results recorded; `sample_info` in every run's meta. |
| 1bd46de | **Bug fix (invalid model name), made before any results were seen:** `scripts/time_one_seed.sh` asked for model `RF`, which the driver does not know (`RandomForest`); the affected parts failed immediately and produced no results. |
| 5047ab7 | Documentation: where design weights are and are not used (section 7), as already implemented. |
| 1f8a17c | **Guard (no protocol change):** `check_benign_weights_constant` (scripts/leakage_study.py) raises if benign `design_weight` varies within a source; called wherever an alert threshold is set (`operational_metrics`, and the gated detector per source). This is the condition under which the unweighted benign quantile equals the weighted one (section 7). Passes on both stratified samples (`tests/test_row_weights.py`). |
| 0a48c0d | **Addition requested before launch, before any result was seen:** secondary response-time metrics (a)–(d) for research objective 2 (section 6), and per-flow prediction files for every run and seed (section 8). No change to the primary metric, thresholds, splits or selection rules. |
| 0a48c0d | **Bug fix (results bookkeeping), made before any results were seen:** `append()` wrote the CSV header from the first batch only, but column sets differ between batches (per-family TPR columns per source; bootstrap columns only for episode protocols), so later rows were misaligned. Affected only the one-seed timing and check CSVs (`results/*_timing_*.csv`, `*_predcheck.csv`), which are not used; the fixed `append()` widens the header and keeps rows aligned (`tests/test_predictions.py`). Saved predictions reproduce the CSV TPR@1%FPR values to < 1e-8 (UNSW-NB15 check run). |
| fe11fd9 | **Addition before launch, before any result was seen:** rule for the best E-GraphSAGE endpoint in the 30-seed runs (validation PR-AUC, real endpoints only; section 6); option (b-ii) and the launch script recorded (section 8). |
| 54d8b3c | **Scope decision (reporting only, no protocol change), made during the run before any result was seen:** response-time metrics (a)–(d) go to paper 3, not paper 2; (a)–(c) computed into `outputs/paper3_baseline/`; prediction files kept permanently. Only this file changed; the run's code is unchanged since e800706. |
| 8d56a87 | **Post-hoc reporting addition (deviation, decided 2026-10-20 after results were seen):** threshold ties. Under the frozen `score > τ` rule, 11 of 197 cells have seeds where τ equals the maximum score (≥ 1% of validation benign flows share it), so no flow can alert (`stats/threshold_ties.csv`). Main text: these cells are shown as "cannot operate at 1% FPR (tied scores)", not 0.000; the primary rule is unchanged. Supplement only: a sensitivity analysis with `score ≥ τ` for these 11 cells, with the FPR it reaches on test (`scripts/tie_sensitivity.py`, `reports/tie_sensitivity_ge.md`). Every table reporting TPR@1%FPR also shows the FPR reached on test. The paper 3 baseline uses the frozen rule and flags tied cells the same way. |
| 2ef366d | **Post-hoc analysis (deviation, decided 2026-10-20 after results were seen):** same-family comparison of subset (i) families under the random-row split, with and without the episode split's family mix, from the saved predictions (HistGB flow, seeds 0-9; `reports/posthoc_same_family.md`). Reported as post-hoc; no primary result changes. |
| ddfc1b0 | **Post-hoc analysis (deviation, decided 2026-10-20 after results were seen):** decomposition of the random-vs-episode gap (C4) into family mix, threshold transfer and remainder, for HistGB, RF and MLP on subset (iii), from the saved predictions (`scripts/posthoc_split_decomposition.py`, `reports/posthoc_split_decomposition.md`). Reported as post-hoc; no primary result changes. |
| 5f824aa | **Statistics implementation fix, made after results were seen:** family F7 (inductive vs transductive E-GraphSAGE) was declared as 3 tests per family (the endpoints within a source and protocol), but the code grouped by endpoint, so no Holm correction was applied. Holm is now applied over the 3 endpoints as declared (`holm_f7` in `scripts/paper2_stats.py`, test added). Only F7 adjusted p-values change, and only upwards; no other result changes. |
| 6ae7230 | **Post-hoc analysis (deviation, decided 2026-10-21 after results were seen):** subset (i) TPR with benign-identical attack flows removed (exploratory Table S4; flags from the v1 audit's hashing, reproduced within 0.005 per family; `scripts/posthoc_ambiguity_removed.py`). Reported as post-hoc; no primary result changes. |
