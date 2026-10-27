# Results summary — leakage audit of graph-based NIDS

Generated 2026-10-14 by `scripts/make_export.py` from the result CSVs in this folder. Every number below is computed from those files.

**Study:** *How Much of Graph-Based Intrusion Detection Performance Is Real? Shortcut, Split and Label Leakage in a Merged NetFlow Benchmark.* An evaluation study of BigFlow-NIDS (66.9M flows), with external validation on the original UNSW-NB15 and on CIC-IoT2023.

## 0. Scope notes (read first)

- **No "adaptive filtering method" exists in this study.** The closest component is the *family-gated detector* (§2.6), which adaptively chooses a feature-set expert per attack family. It is described in full below.
- **Detection/response time, latency and throughput were not measured.** No inference-latency or throughput experiment was run. §5.12 reports only the wall-clock time of each experiment stage, which is not a detection-latency measure.
- **Accuracy** is reported only for the pooled shortcut analysis (§5.1). The main experiments use operational metrics instead: detection rate (TPR) at a fixed 1% / 0.1% false-positive rate, the false-positive rate actually reached on test data, prevalence-weighted precision, F1 and PR-AUC, and ROC-AUC. Accuracy is uninformative at these class imbalances (e.g. 97.7% attacks in CIC-IoT2023).
- `(w)` = weighted so that test flows have each source's real class prevalence (§1.4).
- Values are mean ± sample standard deviation over seeds; `seeds` gives the number of runs.

## 1. Datasets

### 1.1 Overview

| dataset | rows | features used | attack share | timestamps / IPs | notes |
|---|---|---|---|---|---|
| BigFlow-NIDS (Mendeley v2, DOI 10.17632/nv729tbdgz.2) | 66,935,021 | 47 | 0.4533 | yes / yes | merge of NetFlow-v3 UNSW-NB15, CSE-CIC-IDS2018, BoT-IoT, ToN-IoT; 67 files |
| UNSW-NB15 (original, 4 raw CSVs) | 2,540,047 | 41 | 0.1265 | yes / yes | Argus/Bro features; plus official train/test partition |
| CIC-IoT2023 (feature CSVs, 169 files) | 46,686,579 | 46 | 0.9765 | no / no | packet-window features, 34 labels; partial protocol only |

BigFlow by recovered source dataset:

| source | flows | attack_share | distinct_vectors | irreducible_error | attack flows in ambiguous vectors |
|---|---|---|---|---|---|
| BoT-IoT | 16,933,808 | 0.9969 | 1,556,305 | 0.0014 | 0.0224 |
| CSE-CIC-IDS2018 | 20,115,529 | 0.1293 | 9,057,473 | 0.0051 | 0.0576 |
| ToN-IoT | 27,520,260 | 0.3898 | 5,742,209 | 0.0421 | 0.5595 |
| UNSW-NB15 | 2,365,424 | 0.0540 | 1,403,719 | 0.0000 | 0.0000 |

### 1.2 Class distribution (all rows)

BigFlow-NIDS, per source and class:

| source | class | flows | share of source (%) |
|---|---|---|---|
| BoT-IoT | DoS | 8,034,190 | 47.445 |
| BoT-IoT | DDoS | 7,150,882 | 42.228 |
| BoT-IoT | Reconnaissance | 1,695,132 | 10.010 |
| BoT-IoT | Benign | 51,989 | 0.307 |
| BoT-IoT | Theft | 1,615 | 0.010 |
| CSE-CIC-IDS2018 | Benign | 17,514,626 | 87.070 |
| CSE-CIC-IDS2018 | DDOS_attack-HOIC | 1,032,311 | 5.132 |
| CSE-CIC-IDS2018 | FTP-BruteForce | 386,720 | 1.922 |
| CSE-CIC-IDS2018 | DDoS_attacks-LOIC-HTTP | 288,589 | 1.435 |
| CSE-CIC-IDS2018 | Bot | 207,703 | 1.033 |
| CSE-CIC-IDS2018 | SSH-Bruteforce | 188,474 | 0.937 |
| CSE-CIC-IDS2018 | Infilteration | 188,152 | 0.935 |
| CSE-CIC-IDS2018 | DoS_attacks-SlowHTTPTest | 105,550 | 0.525 |
| CSE-CIC-IDS2018 | DoS_attacks-Hulk | 100,076 | 0.498 |
| CSE-CIC-IDS2018 | DoS_attacks-GoldenEye | 61,300 | 0.305 |
| CSE-CIC-IDS2018 | DoS_attacks-Slowloris | 36,040 | 0.179 |
| CSE-CIC-IDS2018 | DDOS_attack-LOIC-UDP | 3,450 | 0.017 |
| CSE-CIC-IDS2018 | Brute_Force_-Web | 1,618 | 0.008 |
| CSE-CIC-IDS2018 | Brute_Force_-XSS | 480 | 0.002 |
| CSE-CIC-IDS2018 | SQL_Injection | 440 | 0.002 |
| ToN-IoT | Benign | 16,792,214 | 61.018 |
| ToN-IoT | DDoS | 4,141,256 | 15.048 |
| ToN-IoT | xss | 2,834,435 | 10.299 |
| ToN-IoT | password | 1,594,777 | 5.795 |
| ToN-IoT | scanning | 1,358,977 | 4.938 |
| ToN-IoT | injection | 381,777 | 1.387 |
| ToN-IoT | DoS | 203,456 | 0.739 |
| ToN-IoT | Backdoor | 203,384 | 0.739 |
| ToN-IoT | mitm | 6,013 | 0.022 |
| ToN-IoT | ransomware | 3,971 | 0.014 |
| UNSW-NB15 | Benign | 2,237,731 | 94.602 |
| UNSW-NB15 | Exploits | 42,748 | 1.807 |
| UNSW-NB15 | Fuzzers | 33,816 | 1.430 |
| UNSW-NB15 | Generic | 19,651 | 0.831 |
| UNSW-NB15 | Reconnaissance | 17,074 | 0.722 |
| UNSW-NB15 | DoS | 5,980 | 0.253 |
| UNSW-NB15 | Backdoor | 4,659 | 0.197 |
| UNSW-NB15 | Shellcode | 2,381 | 0.101 |
| UNSW-NB15 | Analysis | 1,226 | 0.052 |
| UNSW-NB15 | Worms | 158 | 0.007 |

External datasets:

| dataset | class | flows | share (%) |
|---|---|---|---|
| UNSW-NB15 (original) | Benign | 2,218,764 | 87.351 |
| UNSW-NB15 (original) | Generic | 215,481 | 8.483 |
| UNSW-NB15 (original) | Exploits | 44,525 | 1.753 |
| UNSW-NB15 (original) | Fuzzers | 24,246 | 0.955 |
| UNSW-NB15 (original) | DoS | 16,353 | 0.644 |
| UNSW-NB15 (original) | Reconnaissance | 13,987 | 0.551 |
| UNSW-NB15 (original) | Analysis | 2,677 | 0.105 |
| UNSW-NB15 (original) | Backdoor | 2,329 | 0.092 |
| UNSW-NB15 (original) | Shellcode | 1,511 | 0.059 |
| UNSW-NB15 (original) | Worms | 174 | 0.007 |
| CIC-IoT2023 | DDoS-ICMP_Flood | 7,200,504 | 15.423 |
| CIC-IoT2023 | DDoS-UDP_Flood | 5,412,287 | 11.593 |
| CIC-IoT2023 | DDoS-TCP_Flood | 4,497,667 | 9.634 |
| CIC-IoT2023 | DDoS-PSHACK_Flood | 4,094,755 | 8.771 |
| CIC-IoT2023 | DDoS-SYN_Flood | 4,059,190 | 8.695 |
| CIC-IoT2023 | DDoS-RSTFINFlood | 4,045,285 | 8.665 |
| CIC-IoT2023 | DDoS-SynonymousIP_Flood | 3,598,138 | 7.707 |
| CIC-IoT2023 | DoS-UDP_Flood | 3,318,595 | 7.108 |
| CIC-IoT2023 | DoS-TCP_Flood | 2,671,445 | 5.722 |
| CIC-IoT2023 | DoS-SYN_Flood | 2,028,834 | 4.346 |
| CIC-IoT2023 | Benign | 1,098,195 | 2.352 |
| CIC-IoT2023 | Mirai-greeth_flood | 991,866 | 2.125 |
| CIC-IoT2023 | Mirai-udpplain | 890,576 | 1.908 |
| CIC-IoT2023 | Mirai-greip_flood | 751,682 | 1.610 |
| CIC-IoT2023 | DDoS-ICMP_Fragmentation | 452,489 | 0.969 |
| CIC-IoT2023 | MITM-ArpSpoofing | 307,593 | 0.659 |
| CIC-IoT2023 | DDoS-UDP_Fragmentation | 286,925 | 0.615 |
| CIC-IoT2023 | DDoS-ACK_Fragmentation | 285,104 | 0.611 |
| CIC-IoT2023 | DNS_Spoofing | 178,911 | 0.383 |
| CIC-IoT2023 | Recon-HostDiscovery | 134,378 | 0.288 |
| CIC-IoT2023 | Recon-OSScan | 98,259 | 0.210 |
| CIC-IoT2023 | Recon-PortScan | 82,284 | 0.176 |
| CIC-IoT2023 | DoS-HTTP_Flood | 71,864 | 0.154 |
| CIC-IoT2023 | VulnerabilityScan | 37,382 | 0.080 |
| CIC-IoT2023 | DDoS-HTTP_Flood | 28,790 | 0.062 |
| CIC-IoT2023 | DDoS-SlowLoris | 23,426 | 0.050 |
| CIC-IoT2023 | DictionaryBruteForce | 13,064 | 0.028 |
| CIC-IoT2023 | BrowserHijacking | 5,859 | 0.013 |
| CIC-IoT2023 | CommandInjection | 5,409 | 0.012 |
| CIC-IoT2023 | SqlInjection | 5,245 | 0.011 |
| CIC-IoT2023 | XSS | 3,846 | 0.008 |
| CIC-IoT2023 | Backdoor_Malware | 3,218 | 0.007 |
| CIC-IoT2023 | Recon-PingSweep | 2,262 | 0.005 |
| CIC-IoT2023 | Uploading_Attack | 1,252 | 0.003 |

UNSW-NB15 official partition (175,341 train / 82,332 test):

| class | train | test |
|---|---|---|
| Normal | 56,000 | 37,000 |
| Generic | 40,000 | 18,871 |
| Exploits | 33,393 | 11,132 |
| Fuzzers | 18,184 | 6,062 |
| DoS | 12,264 | 4,089 |
| Reconnaissance | 10,491 | 3,496 |
| Analysis | 2,000 | 677 |
| Backdoor | 1,746 | 583 |
| Shellcode | 1,133 | 378 |
| Worms | 130 | 44 |

### 1.3 Preprocessing

- **Source recovery (BigFlow):** each flow is assigned to its source dataset from its start timestamp (UNSW-NB15 < 2016; CSE-CIC-IDS2018 Jan–mid-Apr 2018; BoT-IoT mid-Apr–Dec 2018; ToN-IoT 2019); checked against testbed address ranges on 1.2M flows from 12 files.
- **Behaviour features:** removed flow start/end timestamps, source and destination IP, source port and DNS query id (BigFlow: 47 features remain). UNSW-NB15: removed srcip, sport, dstip, dsport, stime, ltime (41 remain; `proto`, `state`, `service` integer-coded). CIC-IoT2023: all 46 features (no identifiers present).
- **Distance-based inputs** (neighbourhood features): log1p of heavy-tailed non-negative counters (skew > 2), one-hot of destination port / protocol / L7 protocol (top 20 + other; BigFlow only), TCP flag bits, standard scaling — all fitted on training rows only.
- **MLP inputs:** sign(x)·log(1+|x|), then standard scaling (fit on training rows).
- **Working samples:** capped at 20,000 flows per (source, attack class) and 100,000 benign per source (all rows kept for rarer classes). Context experiments subsample each source to 60,000 flows stratified by class.

| dataset | sample flows | attack share |
|---|---|---|
| BigFlow: BoT-IoT | 113,732 | 0.5429 |
| BigFlow: CSE-CIC-IDS2018 | 304,490 | 0.6729 |
| BigFlow: ToN-IoT | 250,157 | 0.5993 |
| BigFlow: UNSW-NB15 | 191,318 | 0.4762 |
| UNSW-NB15 (original) | 196,795 | 0.4940 |
| CIC-IoT2023 | 639,807 | 0.8430 |

### 1.4 Train / validation / test splits

All splits are 70% train / 15% validation / 15% test, per source dataset.

- **Random rows:** stratified by attack class (the common practice).
- **10-minute time blocks:** whole 10-min windows assigned at random to a split, then repaired so every class with ≥ 3 blocks is in all splits.
- **Attack episodes (proposed protocol):** a new episode starts after > 5 min without a flow of that class; benign flows grouped into 30-min blocks; whole episodes assigned to splits. Classes with < 3 episodes fall back to a chronological split (earliest 70% train, latest 15% test) — this affects 20% of ToN-IoT and 33% of CSE-CIC-IDS2018 test flows.
- **UNSW-NB15 official partition:** the published training/testing files; 15% of the training file held out as validation.
- CIC-IoT2023: random rows only (no timestamps).

### 1.5 Leakage-audit steps

1. Source-identifiability test: classifier predicting the source dataset from behaviour features; single-feature tests.
2. Exact-duplicate analysis on **all** rows: hash of each behaviour vector; share of attack flows whose vector also occurs as benign; irreducible error = Σ_v min(benign_v, attack_v) / N.
3. Single-feature shortcut test (external datasets): balanced accuracy of a model on one feature.
4. Timestamp-only control model under every split strategy.
5. Train/test duplicate overlap: share of test rows whose exact vector occurs in training.
6. Operational metrics: alert threshold = (1 − FPR) quantile of **validation** benign scores, applied unchanged to test; the FPR actually reached on test is reported.
7. Prevalence weighting: test (and validation, for the F1 threshold) flows weighted by w = N_{source,class} / n_{source,class} (full-data count / sample count).
8. Neighbourhood label features use training labels only, with each training flow's own label excluded (leave-one-out).

## 2. Models and methods

### 2.1 Gradient boosting (HistGradientBoostingClassifier)

300 iterations, learning rate 0.1, `class_weight='balanced'`, scikit-learn defaults otherwise (early stopping 'auto').

### 2.2 Random forest

200 trees, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced_subsample'`.

### 2.3 MLP

Hidden layers (128, 64), ReLU, Adam (lr 1e-3), L2 alpha 1e-4, batch 512, ≤ 100 epochs, early stopping (10% internal validation, patience 8); minority class oversampled to balance; inputs signed-log + standardised.

### 2.4 Context features

- **Graph-neighbourhood (k-NN, k = 20)** in the transformed feature space. Label-aware: share of attack neighbours, distance-weighted share, nearest-neighbour label, per-family shares (training labels only, leave-one-out). Label-free: distance to 1st and k-th neighbour, mean neighbour distance, local outlier factor, neighbour mean / std / deviation of six volume and TTL columns.
- **Host behaviour (78 features, BigFlow and UNSW-NB15):** for each flow, statistics of its source host, destination host and source–destination pair within 1 s, 10 s and 60 s windows — flow count, distinct peers, distinct destination ports, destination-port entropy and ratio, SYN-without-ACK share, unanswered share, tiny-flow (≤ 2 packets) share, mean log bytes. Computed over all flows of the dataset; no labels used.

### 2.5 E-GraphSAGE (re-implemented)

Flows are directed edges between endpoints carrying the transformed flow features (signed log + standard scaling, fit on train). Node states start as constant 1-vectors. Two layers: message = Linear([h_u ‖ x_e]), mean-aggregated over incident edges in both directions, update h_v = ReLU(Linear([h_v ‖ m_v])), hidden 128, dropout 0.2. Each flow is classified from [z_u ‖ z_v]. Full-batch transductive training (all flows in the graph, loss on training flows), Adam lr 1e-3, weight decay 1e-5, class-weighted cross-entropy, ≤ 300 epochs, early stopping on validation PR-AUC (checked every 5 epochs, patience 30), 8 CPU threads. Endpoint definitions: **IP**, **IP:port**, **random** (each flow gets two private nodes — control with no shared structure). Re-implemented from Lo et al. (NOMS 2022); not the authors' code.

### 2.6 Family-gated detector (the adaptive component)

Motivation: whether context helps differs between datasets (helps CSE-CIC-IDS2018, hurts ToN-IoT) and is unknown in advance.

1. **Experts:** three gradient-boosting detectors on flow, flow + neighbourhood and flow + host features.
2. **Router:** a multiclass gradient-boosting model (200 iterations, balanced class weights, no early stopping) predicting the attack family (incl. Benign) from flow features.
3. **Per-family expert choice (validation only):** for each family, compute each expert's TPR at 1% FPR on validation flows of that family; choose the best, preferring the flow expert on ties; families absent from validation default to the flow expert.
4. **Calibration:** each expert's score is mapped to the fraction of validation benign scores it exceeds (+1e-3 × raw score to break ties), so experts share one scale.
5. **Combination:** final score = Σ_f p(f | x) · score_{expert(f)}(x), where p(f | x) is the router's probability for attack family f, renormalised over attack families.
6. Baseline for the adaptivity: the unweighted mean of the three calibrated experts.

## 3. Hyperparameters and training setup

| setting | value |
|---|---|
| Working-sample caps | 20,000 per attack class, 100,000 benign per source |
| Context-experiment size | 60,000 flows per source (stratified) |
| Split proportions | 70 / 15 / 15 |
| Episode gap / benign block | 5 min (1 and 30 min in sensitivity) / 30 min |
| Time block (block split) | 10 min |
| Min episodes before chronological fallback | 3 |
| k (neighbourhood) | 20 |
| Host windows | 1 s, 10 s, 60 s |
| Target FPRs | 1% and 0.1% |
| Seeds | 10 (context, E-GraphSAGE, gated, gap); 5 (split comparison); 3 (pooled shortcuts) |
| Statistical test | two-sided paired Wilcoxon signed-rank over seeds (min p = 0.002 with 10) |
| HistGB | 300 iter, lr 0.1, balanced weights |
| Random forest | 200 trees, min_samples_leaf 2, sqrt features, balanced_subsample |
| MLP | (128, 64), Adam 1e-3, alpha 1e-4, batch 512, ≤ 100 epochs, early stopping |
| E-GraphSAGE | 2 layers, hidden 128, dropout 0.2, Adam 1e-3, wd 1e-5, ≤ 300 epochs, patience 30 |
| Gated router | HistGB multiclass, 200 iter, no early stopping |

**Environment:**

| item | value |
|---|---|
| CPU | Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz (24 logical cores) |
| Memory | 62 GB RAM |
| GPU | none (all models trained on CPU) |
| OS | Linux 7.0.0-31-generic |
| Python | 3.12.3 |
| scikit-learn | 1.9.0 |
| PyTorch | 2.13.0+cu130 |
| PyTorch Geometric | 2.8.0.post1 |
| pandas | 3.0.3 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |

## 4. Baselines compared

- Flow-only gradient boosting, random forest and MLP (reference for every context comparison).
- Random-row split (reference for leakage comparisons) and 10-minute time-block split.
- Timestamp-only and source-dataset-only models (shortcut controls).
- E-GraphSAGE (published graph IDS) with IP, IP:port and random-node endpoints.
- Unweighted expert ensemble (baseline for the family-gated detector).
- UNSW-NB15 official training/testing partition.

## 5. Results

### 5.1 Shortcuts on pooled BigFlow (random split, 3 seeds)

| Model | seeds | Accuracy | Precision | Recall | F1 | FPR | FNR | ROC_AUC | PR_AUC |
|---|---|---|---|---|---|---|---|---|---|
| source prior only | 3 | 0.5905 ± 0.0000 | 0.5905 ± 0.0000 | 1.0000 ± 0.0000 | 0.7426 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5854 ± 0.0000 | 0.6423 ± 0.0000 |
| timestamp only | 3 | 0.8059 ± 0.0041 | 0.7888 ± 0.0031 | 0.9169 ± 0.0033 | 0.8480 ± 0.0032 | 0.3542 ± 0.0054 | 0.0831 ± 0.0033 | 0.8940 ± 0.0031 | 0.9172 ± 0.0023 |
| behaviour features | 3 | 0.9554 ± 0.0010 | 0.9792 ± 0.0010 | 0.9445 ± 0.0018 | 0.9615 ± 0.0009 | 0.0289 ± 0.0015 | 0.0555 ± 0.0018 | 0.9918 ± 0.0003 | 0.9947 ± 0.0002 |

Source identification from behaviour features: balanced accuracy 0.9937 ± 0.0006.

Best single features for source identification:

| feature | source_balanced_accuracy |
|---|---|
| in_bytes | 0.7192 |
| max_ip_pkt_len | 0.7178 |
| longest_flow_pkt | 0.7178 |
| tcp_win_max_in | 0.6934 |
| src_to_dst_avg_throughput | 0.6508 |
| dst_to_src_second_bytes | 0.6450 |
| min_ttl | 0.6385 |
| max_ttl | 0.6155 |
| flow_duration_milliseconds | 0.5939 |
| l7_proto | 0.5827 |

### 5.2 Label ambiguity per attack family (all rows)

| source | attack | flows | distinct_vectors | share_identical_to_benign |
|---|---|---|---|---|
| BoT-IoT | DoS | 8,034,190 | 354,734 | 0.0000 |
| BoT-IoT | DDoS | 7,150,882 | 287,453 | 0.0000 |
| BoT-IoT | Reconnaissance | 1,695,132 | 898,729 | 0.2231 |
| BoT-IoT | Theft | 1,615 | 448 | 0.0000 |
| CSE-CIC-IDS2018 | DDOS_attack-HOIC | 1,032,311 | 273,698 | 0.0000 |
| CSE-CIC-IDS2018 | FTP-BruteForce | 386,720 | 1 | 0.0000 |
| CSE-CIC-IDS2018 | DDoS_attacks-LOIC-HTTP | 288,589 | 266,133 | 0.0288 |
| CSE-CIC-IDS2018 | Bot | 207,703 | 2,804 | 0.0000 |
| CSE-CIC-IDS2018 | SSH-Bruteforce | 188,474 | 93,980 | 0.0000 |
| CSE-CIC-IDS2018 | Infilteration | 188,152 | 50,463 | 0.7446 |
| CSE-CIC-IDS2018 | DoS_attacks-SlowHTTPTest | 105,550 | 1 | 0.0000 |
| CSE-CIC-IDS2018 | DoS_attacks-Hulk | 100,076 | 100,072 | 0.0000 |
| CSE-CIC-IDS2018 | DoS_attacks-GoldenEye | 61,300 | 27,410 | 0.0000 |
| CSE-CIC-IDS2018 | DoS_attacks-Slowloris | 36,040 | 3,510 | 0.0000 |
| CSE-CIC-IDS2018 | DDOS_attack-LOIC-UDP | 3,450 | 1,984 | 0.0000 |
| CSE-CIC-IDS2018 | Brute_Force_-Web | 1,618 | 1,106 | 0.4129 |
| CSE-CIC-IDS2018 | Brute_Force_-XSS | 480 | 67 | 0.9083 |
| CSE-CIC-IDS2018 | SQL_Injection | 440 | 116 | 0.7818 |
| ToN-IoT | DDoS | 4,141,256 | 469,449 | 0.7538 |
| ToN-IoT | xss | 2,834,435 | 1,582,922 | 0.3576 |
| ToN-IoT | password | 1,594,777 | 741,649 | 0.3158 |
| ToN-IoT | scanning | 1,358,977 | 227,702 | 0.9394 |
| ToN-IoT | injection | 381,777 | 339,576 | 0.0721 |
| ToN-IoT | DoS | 203,456 | 125,979 | 0.2780 |
| ToN-IoT | Backdoor | 203,384 | 20,641 | 0.0124 |
| ToN-IoT | mitm | 6,013 | 4,951 | 0.0770 |
| ToN-IoT | ransomware | 3,971 | 2,830 | 0.0894 |
| UNSW-NB15 | Exploits | 42,748 | 40,424 | 0.0000 |
| UNSW-NB15 | Fuzzers | 33,816 | 26,987 | 0.0000 |
| UNSW-NB15 | Generic | 19,651 | 5,726 | 0.0000 |
| UNSW-NB15 | Reconnaissance | 17,074 | 11,908 | 0.0000 |
| UNSW-NB15 | DoS | 5,980 | 5,360 | 0.0000 |
| UNSW-NB15 | Backdoor | 4,659 | 3,656 | 0.0000 |
| UNSW-NB15 | Shellcode | 2,381 | 2,316 | 0.0000 |
| UNSW-NB15 | Analysis | 1,226 | 1,226 | 0.0000 |
| UNSW-NB15 | Worms | 158 | 146 | 0.0000 |

### 5.3 Split strategies (gradient boosting, 5 seeds)

| source | representation | split | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC | test rows duplicated in train |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT | behaviour | random rows | 5 | 0.9798 ± 0.0005 | 0.9658 ± 0.0008 | 0.0108 ± 0.0019 | 0.9989 ± 0.0001 | 0.9992 ± 0.0001 | 1.0000 ± 0.0000 | 0.9956 ± 0.0002 | 0.4274 |
| BoT-IoT | timestamp only | random rows | 5 | 0.2908 ± 0.0350 | 0.0705 ± 0.0214 | 0.0095 ± 0.0011 | 0.9985 ± 0.0001 | 0.9992 ± 0.0000 | 0.9999 ± 0.0000 | 0.9584 ± 0.0007 | 0.4274 |
| BoT-IoT | behaviour | 10-min time blocks | 5 | 0.8770 ± 0.1917 | 0.8749 ± 0.1908 | 0.0474 ± 0.0866 | 0.9982 ± 0.0009 | 0.9984 ± 0.0009 | 0.9999 ± 0.0001 | 0.9682 ± 0.0421 | 0.3666 |
| BoT-IoT | timestamp only | 10-min time blocks | 5 | 0.3927 ± 0.3344 | 0.2975 ± 0.3705 | 0.0176 ± 0.0093 | 0.9969 ± 0.0000 | 0.9984 ± 0.0001 | 0.9997 ± 0.0002 | 0.8478 ± 0.1109 | 0.3666 |
| BoT-IoT | behaviour | attack episodes | 5 | 0.2238 ± 0.3441 | 0.1676 ± 0.2485 | 0.1025 ± 0.0937 | 0.9981 ± 0.0004 | 0.9953 ± 0.0054 | 0.9981 ± 0.0016 | 0.7237 ± 0.2017 | 0.1337 |
| BoT-IoT | timestamp only | attack episodes | 5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0236 ± 0.0301 | 0.9944 ± 0.0057 | 0.8588 ± 0.3100 | 0.9952 ± 0.0020 | 0.2435 ± 0.2641 | 0.1337 |
| CSE-CIC-IDS2018 | behaviour | random rows | 5 | 0.9543 ± 0.0006 | 0.9490 ± 0.0010 | 0.0095 ± 0.0012 | 0.9956 ± 0.0010 | 0.9710 ± 0.0009 | 0.9752 ± 0.0006 | 0.9856 ± 0.0005 | 0.4634 |
| CSE-CIC-IDS2018 | timestamp only | random rows | 5 | 0.1197 ± 0.0229 | 0.0178 ± 0.0023 | 0.0094 ± 0.0010 | 0.5983 ± 0.0045 | 0.6894 ± 0.0010 | 0.6167 ± 0.0075 | 0.9317 ± 0.0014 | 0.4634 |
| CSE-CIC-IDS2018 | behaviour | 10-min time blocks | 5 | 0.9505 ± 0.0094 | 0.9444 ± 0.0111 | 0.0106 ± 0.0017 | 0.9955 ± 0.0035 | 0.9680 ± 0.0064 | 0.9726 ± 0.0049 | 0.9886 ± 0.0039 | 0.3982 |
| CSE-CIC-IDS2018 | timestamp only | 10-min time blocks | 5 | 0.0968 ± 0.1910 | 0.0952 ± 0.1920 | 0.0089 ± 0.0108 | 0.3279 ± 0.1077 | 0.3116 ± 0.2265 | 0.3875 ± 0.0934 | 0.8448 ± 0.0663 | 0.3982 |
| CSE-CIC-IDS2018 | behaviour | attack episodes | 5 | 0.9532 ± 0.0054 | 0.9062 ± 0.0545 | 0.0109 ± 0.0018 | 0.9974 ± 0.0047 | 0.9302 ± 0.0319 | 0.9712 ± 0.0062 | 0.9526 ± 0.0152 | 0.3706 |
| CSE-CIC-IDS2018 | timestamp only | attack episodes | 5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1480 ± 0.0832 | 0.2035 ± 0.1158 | 0.1564 ± 0.0556 | 0.4092 ± 0.0586 | 0.3706 |
| ToN-IoT | behaviour | random rows | 5 | 0.8222 ± 0.0016 | 0.5364 ± 0.0250 | 0.0104 ± 0.0003 | 0.9458 ± 0.0017 | 0.9295 ± 0.0010 | 0.9778 ± 0.0004 | 0.9852 ± 0.0001 | 0.4891 |
| ToN-IoT | timestamp only | random rows | 5 | 0.1141 ± 0.0062 | 0.0468 ± 0.0057 | 0.0098 ± 0.0009 | 0.5820 ± 0.0103 | 0.7197 ± 0.0036 | 0.7845 ± 0.0023 | 0.9239 ± 0.0015 | 0.4891 |
| ToN-IoT | behaviour | 10-min time blocks | 5 | 0.6945 ± 0.1845 | 0.4586 ± 0.1137 | 0.0095 ± 0.0099 | 0.9547 ± 0.0265 | 0.9288 ± 0.0121 | 0.9715 ± 0.0214 | 0.9879 ± 0.0084 | 0.4539 |
| ToN-IoT | timestamp only | 10-min time blocks | 5 | 0.1003 ± 0.0558 | 0.0281 ± 0.0179 | 0.0140 ± 0.0097 | 0.5608 ± 0.0728 | 0.6630 ± 0.0307 | 0.7226 ± 0.0212 | 0.9004 ± 0.0285 | 0.4539 |
| ToN-IoT | behaviour | attack episodes | 5 | 0.7665 ± 0.0954 | 0.4421 ± 0.0628 | 0.0394 ± 0.0454 | 0.9041 ± 0.1140 | 0.8826 ± 0.0448 | 0.9442 ± 0.0384 | 0.9385 ± 0.0450 | 0.4182 |
| ToN-IoT | timestamp only | attack episodes | 5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0010 ± 0.0014 | 0.3900 ± 0.0394 | 0.5466 ± 0.0364 | 0.4912 ± 0.1198 | 0.5523 ± 0.1426 | 0.4182 |
| UNSW-NB15 | behaviour | random rows | 5 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0072 ± 0.0007 | 0.9998 ± 0.0005 | 0.9994 ± 0.0005 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3291 |
| UNSW-NB15 | timestamp only | random rows | 5 | 0.0320 ± 0.0040 | 0.0008 ± 0.0019 | 0.0105 ± 0.0017 | 0.1160 ± 0.0034 | 0.1878 ± 0.0026 | 0.1164 ± 0.0016 | 0.7666 ± 0.0012 | 0.3291 |
| UNSW-NB15 | behaviour | 10-min time blocks | 5 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0038 ± 0.0031 | 0.9995 ± 0.0012 | 0.9996 ± 0.0006 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3103 |
| UNSW-NB15 | timestamp only | 10-min time blocks | 5 | 0.1245 ± 0.1457 | 0.1245 ± 0.1457 | 0.0650 ± 0.0415 | 0.1040 ± 0.0146 | 0.1809 ± 0.0290 | 0.1081 ± 0.0262 | 0.7550 ± 0.0533 | 0.3103 |
| UNSW-NB15 | behaviour | attack episodes | 5 | 1.0000 ± 0.0000 | 1.0000 ± 0.0001 | 0.0103 ± 0.0042 | 0.9985 ± 0.0021 | 0.9989 ± 0.0011 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.2210 |
| UNSW-NB15 | timestamp only | attack episodes | 5 | 0.0560 ± 0.0804 | 0.0560 ± 0.0804 | 0.1140 ± 0.1609 | 0.0771 ± 0.0182 | 0.1423 ± 0.0322 | 0.0633 ± 0.0242 | 0.4898 ± 0.0748 | 0.2210 |

### 5.4 Split inflation across model families (5 seeds; E-GraphSAGE 10)

| source | model | split | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| ToN-IoT | RandomForest | random rows | 5 | 0.8191 ± 0.0050 | 0.4384 ± 0.0196 | 0.0111 ± 0.0004 | 0.9466 ± 0.0041 | 0.9276 ± 0.0013 | 0.9763 ± 0.0007 | 0.9843 ± 0.0001 |
| ToN-IoT | MLP | random rows | 5 | 0.7965 ± 0.0240 | 0.5171 ± 0.0133 | 0.0112 ± 0.0015 | 0.9419 ± 0.0016 | 0.9250 ± 0.0013 | 0.9744 ± 0.0012 | 0.9834 ± 0.0004 |
| ToN-IoT | RandomForest | attack episodes | 5 | 0.7522 ± 0.1098 | 0.3437 ± 0.1640 | 0.0223 ± 0.0144 | 0.9108 ± 0.1076 | 0.8769 ± 0.0463 | 0.9329 ± 0.0254 | 0.9082 ± 0.0473 |
| ToN-IoT | MLP | attack episodes | 5 | 0.7484 ± 0.1031 | 0.3929 ± 0.0410 | 0.0164 ± 0.0152 | 0.8995 ± 0.1138 | 0.8787 ± 0.0448 | 0.9548 ± 0.0199 | 0.9453 ± 0.0361 |
| CSE-CIC-IDS2018 | RandomForest | random rows | 5 | 0.9546 ± 0.0007 | 0.9479 ± 0.0027 | 0.0107 ± 0.0013 | 0.9923 ± 0.0010 | 0.9703 ± 0.0007 | 0.9751 ± 0.0006 | 0.9851 ± 0.0004 |
| CSE-CIC-IDS2018 | MLP | random rows | 5 | 0.9527 ± 0.0009 | 0.9470 ± 0.0006 | 0.0103 ± 0.0010 | 0.9935 ± 0.0017 | 0.9700 ± 0.0009 | 0.9726 ± 0.0009 | 0.9835 ± 0.0005 |
| CSE-CIC-IDS2018 | RandomForest | attack episodes | 5 | 0.9534 ± 0.0052 | 0.8740 ± 0.0738 | 0.0101 ± 0.0019 | 0.9882 ± 0.0092 | 0.9441 ± 0.0284 | 0.9690 ± 0.0074 | 0.9496 ± 0.0141 |
| CSE-CIC-IDS2018 | MLP | attack episodes | 5 | 0.9507 ± 0.0049 | 0.9071 ± 0.0502 | 0.0113 ± 0.0017 | 0.9913 ± 0.0094 | 0.9489 ± 0.0270 | 0.9679 ± 0.0061 | 0.9437 ± 0.0153 |

### 5.5 Context features (attack-episode split, gradient boosting, 10 seeds)

| source | representation | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT | flow | 10 | 0.3635 ± 0.3849 | 0.3403 ± 0.3814 | 0.1490 ± 0.1330 | 0.9977 ± 0.0005 | 0.9595 ± 0.0651 | 0.9984 ± 0.0013 | 0.7528 ± 0.1663 |
| BoT-IoT | flow + neighbourhood | 10 | 0.5475 ± 0.4072 | 0.4911 ± 0.3842 | 0.1675 ± 0.1229 | 0.9978 ± 0.0005 | 0.9986 ± 0.0005 | 0.9988 ± 0.0010 | 0.8297 ± 0.1202 |
| BoT-IoT | flow + host | 10 | 0.7483 ± 0.4000 | 0.7457 ± 0.4007 | 0.1279 ± 0.1202 | 0.9980 ± 0.0005 | 0.9987 ± 0.0005 | 0.9997 ± 0.0005 | 0.9540 ± 0.0632 |
| BoT-IoT | flow + neighbourhood + host | 10 | 0.7556 ± 0.3966 | 0.7531 ± 0.3967 | 0.1526 ± 0.1227 | 0.9979 ± 0.0006 | 0.9985 ± 0.0008 | 0.9995 ± 0.0006 | 0.9223 ± 0.0745 |
| CSE-CIC-IDS2018 | flow | 10 | 0.9187 ± 0.0480 | 0.8000 ± 0.0465 | 0.0100 ± 0.0032 | 0.9890 ± 0.0045 | 0.9210 ± 0.0270 | 0.9577 ± 0.0226 | 0.9630 ± 0.0288 |
| CSE-CIC-IDS2018 | flow + neighbourhood | 10 | 0.9493 ± 0.0104 | 0.9348 ± 0.0145 | 0.0124 ± 0.0062 | 0.9922 ± 0.0047 | 0.9647 ± 0.0039 | 0.9709 ± 0.0064 | 0.9786 ± 0.0077 |
| CSE-CIC-IDS2018 | flow + host | 10 | 0.9673 ± 0.0097 | 0.9401 ± 0.0275 | 0.0115 ± 0.0037 | 0.9744 ± 0.0312 | 0.9579 ± 0.0162 | 0.9882 ± 0.0030 | 0.9935 ± 0.0028 |
| CSE-CIC-IDS2018 | flow + neighbourhood + host | 10 | 0.9686 ± 0.0061 | 0.9504 ± 0.0080 | 0.0126 ± 0.0050 | 0.9847 ± 0.0180 | 0.9668 ± 0.0094 | 0.9882 ± 0.0030 | 0.9934 ± 0.0031 |
| ToN-IoT | flow | 10 | 0.8245 ± 0.0515 | 0.5328 ± 0.1323 | 0.0353 ± 0.0578 | 0.9332 ± 0.0844 | 0.8927 ± 0.0362 | 0.9583 ± 0.0218 | 0.9397 ± 0.0386 |
| ToN-IoT | flow + neighbourhood | 10 | 0.7788 ± 0.1138 | 0.5304 ± 0.1168 | 0.0315 ± 0.0535 | 0.9348 ± 0.0798 | 0.8954 ± 0.0372 | 0.9538 ± 0.0211 | 0.9403 ± 0.0328 |
| ToN-IoT | flow + host | 10 | 0.7364 ± 0.1467 | 0.4748 ± 0.2189 | 0.0525 ± 0.0898 | 0.9358 ± 0.0436 | 0.8871 ± 0.0191 | 0.9319 ± 0.0175 | 0.9060 ± 0.0391 |
| ToN-IoT | flow + neighbourhood + host | 10 | 0.7702 ± 0.1101 | 0.5173 ± 0.1721 | 0.0963 ± 0.2241 | 0.8871 ± 0.1243 | 0.8628 ± 0.0571 | 0.9313 ± 0.0186 | 0.9012 ± 0.0435 |
| UNSW-NB15 | flow | 10 | 1.0000 ± 0.0001 | 0.9999 ± 0.0002 | 0.0116 ± 0.0086 | 0.9940 ± 0.0059 | 0.9896 ± 0.0205 | 0.9986 ± 0.0037 | 1.0000 ± 0.0001 |
| UNSW-NB15 | flow + neighbourhood | 10 | 1.0000 ± 0.0000 | 0.9997 ± 0.0005 | 0.0098 ± 0.0045 | 0.9965 ± 0.0032 | 0.9968 ± 0.0017 | 0.9999 ± 0.0001 | 1.0000 ± 0.0000 |
| UNSW-NB15 | flow + host | 10 | 1.0000 ± 0.0000 | 0.9999 ± 0.0002 | 0.0115 ± 0.0023 | 0.9983 ± 0.0042 | 0.9982 ± 0.0023 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| UNSW-NB15 | flow + neighbourhood + host | 10 | 1.0000 ± 0.0000 | 0.9999 ± 0.0001 | 0.0142 ± 0.0061 | 0.9989 ± 0.0017 | 0.9988 ± 0.0010 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

Per-family TPR at 1% FPR (mean over 10 seeds):

| ToN-IoT family | flow | flow + host | flow + neighbourhood | flow + neighbourhood + host |
|---|---|---|---|---|
| DDoS | 0.9882 | 0.9160 | 0.9674 | 0.9669 |
| DoS | 0.9407 | 0.8832 | 0.9291 | 0.9813 |
| Backdoor | 0.7006 | 0.7238 | 0.8062 | 0.7680 |
| injection | 0.9608 | 0.8840 | 0.9399 | 0.9298 |
| mitm | 0.7896 | 0.5613 | 0.6973 | 0.6557 |
| password | 0.8189 | 0.7278 | 0.7896 | 0.7610 |
| ransomware | 0.0767 | 0.2439 | 0.0707 | 0.1225 |
| scanning | 0.0711 | 0.0084 | 0.0365 | 0.0058 |
| xss | 0.9331 | 0.7996 | 0.8199 | 0.8193 |

| CSE-CIC-IDS2018 family | flow | flow + host | flow + neighbourhood | flow + neighbourhood + host |
|---|---|---|---|---|
| Bot | 0.9997 | 0.9985 | 0.9997 | 0.9986 |
| Brute_Force_-Web | 0.9093 | 0.7801 | 0.9057 | 0.8806 |
| Brute_Force_-XSS | 0.9938 | 0.9822 | 0.9938 | 0.9938 |
| DDOS_attack-HOIC | 0.9998 | 1.0000 | 0.9978 | 1.0000 |
| DDOS_attack-LOIC-UDP | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DDoS_attacks-LOIC-HTTP | 0.7031 | 0.9818 | 0.9878 | 0.9947 |
| DoS_attacks-GoldenEye | 0.9735 | 1.0000 | 0.9726 | 1.0000 |
| DoS_attacks-Hulk | 0.9990 | 1.0000 | 1.0000 | 1.0000 |
| DoS_attacks-SlowHTTPTest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DoS_attacks-Slowloris | 0.9242 | 1.0000 | 0.9514 | 1.0000 |
| FTP-BruteForce | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Infilteration | 0.3571 | 0.5796 | 0.3494 | 0.5763 |
| SQL_Injection | 0.9533 | 0.8628 | 0.9460 | 0.9422 |
| SSH-Bruteforce | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Significance vs flow features:

| source | representation | mean diff vs baseline | seeds better | seeds | Wilcoxon p |
|---|---|---|---|---|---|
| BoT-IoT | flow + host | 0.3848 | 9 | 10 | 0.0098 |
| BoT-IoT | flow + neighbourhood | 0.1839 | 8 | 10 | 0.0371 |
| BoT-IoT | flow + neighbourhood + host | 0.3921 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | flow + host | 0.0486 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | flow + neighbourhood | 0.0306 | 6 | 10 | 0.1602 |
| CSE-CIC-IDS2018 | flow + neighbourhood + host | 0.0499 | 10 | 10 | 0.0020 |
| ToN-IoT | flow + host | -0.0881 | 2 | 10 | 0.0195 |
| ToN-IoT | flow + neighbourhood | -0.0457 | 4 | 10 | 0.2324 |
| ToN-IoT | flow + neighbourhood + host | -0.0543 | 5 | 10 | 0.2754 |
| UNSW-NB15 | flow + host | 0.0000 | 1 | 10 | 1.0000 |
| UNSW-NB15 | flow + neighbourhood | 0.0000 | 1 | 10 | 1.0000 |
| UNSW-NB15 | flow + neighbourhood + host | 0.0000 | 1 | 10 | 1.0000 |

### 5.6 Context features across model families (10 seeds)

| source | model | representation | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| ToN-IoT | RandomForest | flow | 10 | 0.8001 ± 0.0518 | 0.4997 ± 0.2038 | 0.0350 ± 0.0592 | 0.9339 ± 0.0879 | 0.8808 ± 0.0362 | 0.9480 ± 0.0230 | 0.9375 ± 0.0440 |
| ToN-IoT | RandomForest | flow + neighbourhood | 10 | 0.7861 ± 0.0772 | 0.4236 ± 0.2354 | 0.0327 ± 0.0543 | 0.9376 ± 0.0800 | 0.8876 ± 0.0343 | 0.9494 ± 0.0296 | 0.9433 ± 0.0440 |
| ToN-IoT | RandomForest | flow + host | 10 | 0.7644 ± 0.1304 | 0.3594 ± 0.2712 | 0.0968 ± 0.2339 | 0.9186 ± 0.1381 | 0.8753 ± 0.0699 | 0.9262 ± 0.0161 | 0.9175 ± 0.0391 |
| ToN-IoT | RandomForest | flow + neighbourhood + host | 10 | 0.7963 ± 0.1079 | 0.4429 ± 0.2795 | 0.0966 ± 0.2330 | 0.8804 ± 0.1692 | 0.8576 ± 0.0890 | 0.9340 ± 0.0178 | 0.9206 ± 0.0453 |
| ToN-IoT | MLP | flow | 10 | 0.7606 ± 0.1019 | 0.3785 ± 0.2092 | 0.0184 ± 0.0103 | 0.9272 ± 0.0804 | 0.8874 ± 0.0305 | 0.9538 ± 0.0200 | 0.9494 ± 0.0262 |
| ToN-IoT | MLP | flow + neighbourhood | 10 | 0.7769 ± 0.1278 | 0.4668 ± 0.1695 | 0.0183 ± 0.0142 | 0.9287 ± 0.0818 | 0.8956 ± 0.0383 | 0.9586 ± 0.0189 | 0.9462 ± 0.0239 |
| ToN-IoT | MLP | flow + host | 10 | 0.7375 ± 0.0946 | 0.4337 ± 0.2089 | 0.0338 ± 0.0319 | 0.9077 ± 0.0890 | 0.8757 ± 0.0461 | 0.9159 ± 0.0291 | 0.8952 ± 0.0523 |
| ToN-IoT | MLP | flow + neighbourhood + host | 10 | 0.7469 ± 0.0966 | 0.4681 ± 0.1811 | 0.0433 ± 0.0562 | 0.8659 ± 0.1190 | 0.8603 ± 0.0667 | 0.9235 ± 0.0301 | 0.9032 ± 0.0461 |
| CSE-CIC-IDS2018 | RandomForest | flow | 10 | 0.9347 ± 0.0380 | 0.7615 ± 0.0638 | 0.0091 ± 0.0023 | 0.9860 ± 0.0069 | 0.9115 ± 0.0194 | 0.9666 ± 0.0060 | 0.9767 ± 0.0069 |
| CSE-CIC-IDS2018 | RandomForest | flow + neighbourhood | 10 | 0.9500 ± 0.0094 | 0.8517 ± 0.0521 | 0.0108 ± 0.0042 | 0.9801 ± 0.0321 | 0.9273 ± 0.0275 | 0.9699 ± 0.0053 | 0.9793 ± 0.0056 |
| CSE-CIC-IDS2018 | RandomForest | flow + host | 10 | 0.9667 ± 0.0069 | 0.8839 ± 0.0460 | 0.0108 ± 0.0022 | 0.9826 ± 0.0116 | 0.9450 ± 0.0299 | 0.9874 ± 0.0032 | 0.9935 ± 0.0030 |
| CSE-CIC-IDS2018 | RandomForest | flow + neighbourhood + host | 10 | 0.9656 ± 0.0069 | 0.9478 ± 0.0107 | 0.0116 ± 0.0036 | 0.9888 ± 0.0064 | 0.9680 ± 0.0035 | 0.9877 ± 0.0032 | 0.9937 ± 0.0029 |
| CSE-CIC-IDS2018 | MLP | flow | 10 | 0.9442 ± 0.0095 | 0.8453 ± 0.0746 | 0.0092 ± 0.0028 | 0.9848 ± 0.0073 | 0.9471 ± 0.0148 | 0.9655 ± 0.0055 | 0.9715 ± 0.0071 |
| CSE-CIC-IDS2018 | MLP | flow + neighbourhood | 10 | 0.9497 ± 0.0090 | 0.9378 ± 0.0114 | 0.0109 ± 0.0051 | 0.9919 ± 0.0046 | 0.9656 ± 0.0056 | 0.9703 ± 0.0055 | 0.9755 ± 0.0088 |
| CSE-CIC-IDS2018 | MLP | flow + host | 10 | 0.9615 ± 0.0269 | 0.9212 ± 0.0452 | 0.0115 ± 0.0045 | 0.9852 ± 0.0127 | 0.9583 ± 0.0209 | 0.9865 ± 0.0068 | 0.9911 ± 0.0058 |
| CSE-CIC-IDS2018 | MLP | flow + neighbourhood + host | 10 | 0.9689 ± 0.0057 | 0.9507 ± 0.0086 | 0.0121 ± 0.0032 | 0.9811 ± 0.0156 | 0.9646 ± 0.0085 | 0.9883 ± 0.0028 | 0.9932 ± 0.0029 |

| source | model | representation | mean diff vs baseline | seeds better | seeds | Wilcoxon p |
|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | MLP | flow + host | 0.0173 | 9 | 10 | 0.0840 |
| CSE-CIC-IDS2018 | MLP | flow + neighbourhood | 0.0055 | 9 | 10 | 0.0039 |
| CSE-CIC-IDS2018 | MLP | flow + neighbourhood + host | 0.0247 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | RandomForest | flow + host | 0.0320 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | RandomForest | flow + neighbourhood | 0.0154 | 6 | 10 | 0.0840 |
| CSE-CIC-IDS2018 | RandomForest | flow + neighbourhood + host | 0.0309 | 10 | 10 | 0.0020 |
| ToN-IoT | MLP | flow + host | -0.0231 | 4 | 10 | 0.4316 |
| ToN-IoT | MLP | flow + neighbourhood | 0.0163 | 8 | 10 | 0.1934 |
| ToN-IoT | MLP | flow + neighbourhood + host | -0.0137 | 5 | 10 | 0.6953 |
| ToN-IoT | RandomForest | flow + host | -0.0357 | 7 | 10 | 1.0000 |
| ToN-IoT | RandomForest | flow + neighbourhood | -0.0140 | 4 | 10 | 0.3223 |
| ToN-IoT | RandomForest | flow + neighbourhood + host | -0.0038 | 7 | 10 | 0.6250 |

### 5.7 E-GraphSAGE (10 seeds)

| source | split | model | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| ToN-IoT | random rows | HistGB (flow, no graph) | 10 | 0.8158 ± 0.0249 | 0.4921 ± 0.1003 | 0.0101 ± 0.0019 | 0.9413 ± 0.0053 | 0.9289 ± 0.0015 | 0.9775 ± 0.0011 | 0.9849 ± 0.0006 |
| ToN-IoT | random rows | E-GraphSAGE (ip) | 10 | 0.5866 ± 0.0074 | 0.3637 ± 0.1312 | 0.0105 ± 0.0018 | 0.7727 ± 0.0368 | 0.8255 ± 0.0037 | 0.9134 ± 0.0036 | 0.9457 ± 0.0017 |
| ToN-IoT | random rows | E-GraphSAGE (ip_port) | 10 | 0.7491 ± 0.0374 | 0.3866 ± 0.0581 | 0.0107 ± 0.0021 | 0.9425 ± 0.0066 | 0.9228 ± 0.0017 | 0.9728 ± 0.0012 | 0.9821 ± 0.0006 |
| ToN-IoT | random rows | E-GraphSAGE (random) | 10 | 0.7955 ± 0.0190 | 0.4309 ± 0.0837 | 0.0105 ± 0.0022 | 0.9371 ± 0.0036 | 0.9242 ± 0.0014 | 0.9749 ± 0.0011 | 0.9835 ± 0.0004 |
| ToN-IoT | attack episodes | HistGB (flow, no graph) | 10 | 0.8245 ± 0.0515 | 0.5328 ± 0.1323 | 0.0353 ± 0.0578 | 0.9332 ± 0.0844 | 0.8927 ± 0.0362 | 0.9583 ± 0.0218 | 0.9397 ± 0.0386 |
| ToN-IoT | attack episodes | E-GraphSAGE (ip) | 10 | 0.6226 ± 0.2028 | 0.3120 ± 0.1786 | 0.0164 ± 0.0157 | 0.7397 ± 0.1241 | 0.7674 ± 0.0862 | 0.8770 ± 0.0974 | 0.8732 ± 0.0633 |
| ToN-IoT | attack episodes | E-GraphSAGE (ip_port) | 10 | 0.7189 ± 0.0982 | 0.3513 ± 0.1627 | 0.0184 ± 0.0133 | 0.9253 ± 0.0505 | 0.8886 ± 0.0292 | 0.9400 ± 0.0212 | 0.9214 ± 0.0281 |
| ToN-IoT | attack episodes | E-GraphSAGE (random) | 10 | 0.6958 ± 0.1179 | 0.2741 ± 0.1785 | 0.0191 ± 0.0101 | 0.8969 ± 0.0846 | 0.8791 ± 0.0340 | 0.9402 ± 0.0284 | 0.9343 ± 0.0431 |
| CSE-CIC-IDS2018 | random rows | HistGB (flow, no graph) | 10 | 0.9521 ± 0.0013 | 0.9452 ± 0.0057 | 0.0097 ± 0.0025 | 0.9928 ± 0.0037 | 0.9691 ± 0.0016 | 0.9737 ± 0.0006 | 0.9845 ± 0.0004 |
| CSE-CIC-IDS2018 | random rows | E-GraphSAGE (ip) | 10 | 0.9635 ± 0.0063 | 0.9440 ± 0.0043 | 0.0097 ± 0.0022 | 0.9935 ± 0.0058 | 0.9679 ± 0.0016 | 0.9945 ± 0.0005 | 0.9988 ± 0.0001 |
| CSE-CIC-IDS2018 | random rows | E-GraphSAGE (ip_port) | 10 | 0.9535 ± 0.0015 | 0.9410 ± 0.0064 | 0.0096 ± 0.0022 | 0.9909 ± 0.0045 | 0.9670 ± 0.0017 | 0.9736 ± 0.0013 | 0.9846 ± 0.0007 |
| CSE-CIC-IDS2018 | random rows | E-GraphSAGE (random) | 10 | 0.9508 ± 0.0011 | 0.9414 ± 0.0059 | 0.0099 ± 0.0025 | 0.9928 ± 0.0053 | 0.9680 ± 0.0020 | 0.9713 ± 0.0024 | 0.9831 ± 0.0005 |
| CSE-CIC-IDS2018 | attack episodes | HistGB (flow, no graph) | 10 | 0.9187 ± 0.0480 | 0.8000 ± 0.0465 | 0.0100 ± 0.0032 | 0.9890 ± 0.0045 | 0.9210 ± 0.0270 | 0.9577 ± 0.0226 | 0.9630 ± 0.0288 |
| CSE-CIC-IDS2018 | attack episodes | E-GraphSAGE (ip) | 10 | 0.9654 ± 0.0098 | 0.9458 ± 0.0140 | 0.0102 ± 0.0056 | 0.9845 ± 0.0197 | 0.9618 ± 0.0063 | 0.9942 ± 0.0014 | 0.9987 ± 0.0003 |
| CSE-CIC-IDS2018 | attack episodes | E-GraphSAGE (ip_port) | 10 | 0.9546 ± 0.0087 | 0.9389 ± 0.0085 | 0.0105 ± 0.0045 | 0.9922 ± 0.0052 | 0.9680 ± 0.0054 | 0.9740 ± 0.0055 | 0.9836 ± 0.0051 |
| CSE-CIC-IDS2018 | attack episodes | E-GraphSAGE (random) | 10 | 0.9354 ± 0.0196 | 0.8351 ± 0.0794 | 0.0103 ± 0.0022 | 0.9899 ± 0.0044 | 0.9300 ± 0.0275 | 0.9613 ± 0.0111 | 0.9651 ± 0.0138 |

| source | split | model | mean diff vs baseline | seeds better | seeds | Wilcoxon p |
|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | attack episodes | E-GraphSAGE (ip) | 0.0467 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | attack episodes | E-GraphSAGE (ip_port) | 0.0359 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | attack episodes | E-GraphSAGE (random) | 0.0168 | 4 | 10 | 1.0000 |
| CSE-CIC-IDS2018 | random rows | E-GraphSAGE (ip) | 0.0114 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | random rows | E-GraphSAGE (ip_port) | 0.0014 | 9 | 10 | 0.0039 |
| CSE-CIC-IDS2018 | random rows | E-GraphSAGE (random) | -0.0013 | 1 | 10 | 0.0098 |
| ToN-IoT | attack episodes | E-GraphSAGE (ip) | -0.2019 | 0 | 10 | 0.0020 |
| ToN-IoT | attack episodes | E-GraphSAGE (ip_port) | -0.1056 | 1 | 10 | 0.0039 |
| ToN-IoT | attack episodes | E-GraphSAGE (random) | -0.1287 | 1 | 10 | 0.0059 |
| ToN-IoT | random rows | E-GraphSAGE (ip) | -0.2292 | 0 | 10 | 0.0020 |
| ToN-IoT | random rows | E-GraphSAGE (ip_port) | -0.0667 | 0 | 10 | 0.0020 |
| ToN-IoT | random rows | E-GraphSAGE (random) | -0.0203 | 2 | 10 | 0.0488 |

### 5.8 Family-gated detector (10 seeds)

| source | representation | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| ToN-IoT | flow | 10 | 0.8245 ± 0.0515 | 0.5328 ± 0.1323 | 0.0353 ± 0.0578 | 0.9332 ± 0.0844 | 0.8927 ± 0.0362 | 0.9583 ± 0.0218 | 0.9397 ± 0.0386 |
| ToN-IoT | flow + neighbourhood | 10 | 0.7788 ± 0.1138 | 0.5304 ± 0.1168 | 0.0315 ± 0.0535 | 0.9348 ± 0.0798 | 0.8954 ± 0.0372 | 0.9538 ± 0.0211 | 0.9403 ± 0.0328 |
| ToN-IoT | flow + host | 10 | 0.7364 ± 0.1467 | 0.4748 ± 0.2189 | 0.0525 ± 0.0898 | 0.9358 ± 0.0436 | 0.8871 ± 0.0191 | 0.9319 ± 0.0175 | 0.9060 ± 0.0391 |
| ToN-IoT | ensemble (mean of experts) | 10 | 0.8136 ± 0.0752 | 0.5735 ± 0.1760 | 0.0195 ± 0.0123 | 0.9224 ± 0.1302 | 0.8816 ± 0.0631 | 0.9512 ± 0.0192 | 0.9327 ± 0.0315 |
| ToN-IoT | family-gated (ours) | 10 | 0.8291 ± 0.0566 | 0.5617 ± 0.1393 | 0.0356 ± 0.0580 | 0.9357 ± 0.0861 | 0.8954 ± 0.0380 | 0.9506 ± 0.0201 | 0.9284 ± 0.0341 |
| CSE-CIC-IDS2018 | flow | 10 | 0.9187 ± 0.0480 | 0.8000 ± 0.0465 | 0.0100 ± 0.0032 | 0.9890 ± 0.0045 | 0.9210 ± 0.0270 | 0.9577 ± 0.0226 | 0.9630 ± 0.0288 |
| CSE-CIC-IDS2018 | flow + neighbourhood | 10 | 0.9493 ± 0.0104 | 0.9348 ± 0.0145 | 0.0124 ± 0.0062 | 0.9922 ± 0.0047 | 0.9647 ± 0.0039 | 0.9709 ± 0.0064 | 0.9786 ± 0.0077 |
| CSE-CIC-IDS2018 | flow + host | 10 | 0.9673 ± 0.0097 | 0.9401 ± 0.0275 | 0.0115 ± 0.0037 | 0.9744 ± 0.0312 | 0.9579 ± 0.0162 | 0.9882 ± 0.0030 | 0.9935 ± 0.0028 |
| CSE-CIC-IDS2018 | ensemble (mean of experts) | 10 | 0.9456 ± 0.0177 | 0.9207 ± 0.0460 | 0.0106 ± 0.0032 | 0.9920 ± 0.0124 | 0.9354 ± 0.0242 | 0.9767 ± 0.0048 | 0.9859 ± 0.0048 |
| CSE-CIC-IDS2018 | family-gated (ours) | 10 | 0.9661 ± 0.0060 | 0.8659 ± 0.0495 | 0.0106 ± 0.0034 | 0.9751 ± 0.0288 | 0.9473 ± 0.0215 | 0.9863 ± 0.0028 | 0.9916 ± 0.0036 |

| source | representation | mean diff vs baseline | seeds better | seeds | Wilcoxon p | baseline |
|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | family-gated (ours) | 0.0474 | 10 | 10 | 0.0020 | flow |
| ToN-IoT | family-gated (ours) | 0.0046 | 6 | 10 | 0.1934 | flow |
| CSE-CIC-IDS2018 | family-gated (ours) | -0.0012 | 1 | 10 | 0.0840 | flow + host |
| ToN-IoT | family-gated (ours) | 0.0927 | 9 | 10 | 0.0039 | flow + host |
| CSE-CIC-IDS2018 | family-gated (ours) | 0.0205 | 10 | 10 | 0.0020 | ensemble (mean of experts) |
| ToN-IoT | family-gated (ours) | 0.0155 | 6 | 10 | 0.2754 | ensemble (mean of experts) |

Expert chosen per family (count over seeds):

| source | family | flow | flow + host | flow + neighbourhood |
|---|---|---|---|---|
| CSE-CIC-IDS2018 | Bot | 10 | 0 | 0 |
| CSE-CIC-IDS2018 | Brute_Force_-Web | 7 | 1 | 2 |
| CSE-CIC-IDS2018 | Brute_Force_-XSS | 8 | 2 | 0 |
| CSE-CIC-IDS2018 | DDOS_attack-HOIC | 10 | 0 | 0 |
| CSE-CIC-IDS2018 | DDOS_attack-LOIC-UDP | 10 | 0 | 0 |
| CSE-CIC-IDS2018 | DDoS_attacks-LOIC-HTTP | 4 | 1 | 1 |
| CSE-CIC-IDS2018 | DoS_attacks-GoldenEye | 7 | 3 | 0 |
| CSE-CIC-IDS2018 | DoS_attacks-Hulk | 10 | 0 | 0 |
| CSE-CIC-IDS2018 | DoS_attacks-SlowHTTPTest | 10 | 0 | 0 |
| CSE-CIC-IDS2018 | DoS_attacks-Slowloris | 10 | 0 | 0 |
| CSE-CIC-IDS2018 | FTP-BruteForce | 10 | 0 | 0 |
| CSE-CIC-IDS2018 | Infilteration | 0 | 10 | 0 |
| CSE-CIC-IDS2018 | SQL_Injection | 9 | 1 | 0 |
| CSE-CIC-IDS2018 | SSH-Bruteforce | 10 | 0 | 0 |
| ToN-IoT | Backdoor | 3 | 2 | 3 |
| ToN-IoT | DDoS | 8 | 1 | 1 |
| ToN-IoT | DoS | 1 | 8 | 1 |
| ToN-IoT | injection | 3 | 3 | 2 |
| ToN-IoT | mitm | 6 | 1 | 1 |
| ToN-IoT | password | 6 | 3 | 1 |
| ToN-IoT | ransomware | 2 | 1 | 0 |
| ToN-IoT | scanning | 8 | 1 | 1 |
| ToN-IoT | xss | 5 | 4 | 1 |

### 5.9 Episode-gap sensitivity (gradient boosting, 10 seeds)

| source | gap_min | representation | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| ToN-IoT | 1 | flow | 10 | 0.6393 ± 0.0623 | 0.4726 ± 0.0978 | 0.0212 ± 0.0180 | 0.8617 ± 0.0956 | 0.8257 ± 0.0449 | 0.9215 ± 0.0433 | 0.9165 ± 0.0520 |
| ToN-IoT | 1 | flow + host | 10 | 0.5373 ± 0.0820 | 0.4263 ± 0.1169 | 0.0208 ± 0.0223 | 0.8567 ± 0.1006 | 0.7387 ± 0.0471 | 0.8559 ± 0.0488 | 0.8513 ± 0.0621 |
| ToN-IoT | 1 | flow + neighbourhood + host | 10 | 0.5785 ± 0.0889 | 0.4793 ± 0.1156 | 0.0239 ± 0.0298 | 0.8425 ± 0.1292 | 0.7347 ± 0.0546 | 0.8663 ± 0.0490 | 0.8672 ± 0.0620 |
| ToN-IoT | 5 | flow | 10 | 0.8245 ± 0.0515 | 0.5328 ± 0.1323 | 0.0353 ± 0.0578 | 0.9332 ± 0.0844 | 0.8927 ± 0.0362 | 0.9583 ± 0.0218 | 0.9397 ± 0.0386 |
| ToN-IoT | 5 | flow + host | 10 | 0.7364 ± 0.1467 | 0.4748 ± 0.2189 | 0.0525 ± 0.0898 | 0.9358 ± 0.0436 | 0.8871 ± 0.0191 | 0.9319 ± 0.0175 | 0.9060 ± 0.0391 |
| ToN-IoT | 5 | flow + neighbourhood + host | 10 | 0.7702 ± 0.1101 | 0.5173 ± 0.1721 | 0.0963 ± 0.2241 | 0.8871 ± 0.1243 | 0.8628 ± 0.0571 | 0.9313 ± 0.0186 | 0.9012 ± 0.0435 |
| ToN-IoT | 30 | flow | 10 | 0.7674 ± 0.1191 | 0.4731 ± 0.2019 | 0.0173 ± 0.0137 | 0.9477 ± 0.0282 | 0.9012 ± 0.0190 | 0.9452 ± 0.0245 | 0.8904 ± 0.0600 |
| ToN-IoT | 30 | flow + host | 10 | 0.6863 ± 0.2048 | 0.3987 ± 0.1912 | 0.0398 ± 0.0440 | 0.8784 ± 0.0862 | 0.8579 ± 0.0448 | 0.9011 ± 0.0284 | 0.7890 ± 0.0533 |
| ToN-IoT | 30 | flow + neighbourhood + host | 10 | 0.6940 ± 0.2063 | 0.4208 ± 0.2137 | 0.0600 ± 0.0664 | 0.8206 ± 0.1564 | 0.8262 ± 0.0979 | 0.9040 ± 0.0236 | 0.7857 ± 0.0601 |
| CSE-CIC-IDS2018 | 1 | flow | 10 | 0.9416 ± 0.0193 | 0.7660 ± 0.0686 | 0.0095 ± 0.0027 | 0.9731 ± 0.0174 | 0.9508 ± 0.0181 | 0.9667 ± 0.0060 | 0.9812 ± 0.0060 |
| CSE-CIC-IDS2018 | 1 | flow + host | 10 | 0.9412 ± 0.0247 | 0.8689 ± 0.0660 | 0.0119 ± 0.0047 | 0.9710 ± 0.0230 | 0.9338 ± 0.0377 | 0.9817 ± 0.0063 | 0.9914 ± 0.0032 |
| CSE-CIC-IDS2018 | 1 | flow + neighbourhood + host | 10 | 0.9659 ± 0.0110 | 0.9467 ± 0.0135 | 0.0122 ± 0.0056 | 0.9747 ± 0.0260 | 0.9583 ± 0.0163 | 0.9875 ± 0.0042 | 0.9953 ± 0.0021 |
| CSE-CIC-IDS2018 | 5 | flow | 10 | 0.9187 ± 0.0480 | 0.8000 ± 0.0465 | 0.0100 ± 0.0032 | 0.9890 ± 0.0045 | 0.9210 ± 0.0270 | 0.9577 ± 0.0226 | 0.9630 ± 0.0288 |
| CSE-CIC-IDS2018 | 5 | flow + host | 10 | 0.9673 ± 0.0097 | 0.9401 ± 0.0275 | 0.0115 ± 0.0037 | 0.9744 ± 0.0312 | 0.9579 ± 0.0162 | 0.9882 ± 0.0030 | 0.9935 ± 0.0028 |
| CSE-CIC-IDS2018 | 5 | flow + neighbourhood + host | 10 | 0.9686 ± 0.0061 | 0.9504 ± 0.0080 | 0.0126 ± 0.0050 | 0.9847 ± 0.0180 | 0.9668 ± 0.0094 | 0.9882 ± 0.0030 | 0.9934 ± 0.0031 |
| CSE-CIC-IDS2018 | 30 | flow | 10 | 0.9435 ± 0.0021 | 0.8867 ± 0.0405 | 0.0084 ± 0.0027 | 0.9870 ± 0.0150 | 0.9525 ± 0.0127 | 0.9686 ± 0.0012 | 0.9820 ± 0.0007 |
| CSE-CIC-IDS2018 | 30 | flow + host | 10 | 0.9699 ± 0.0035 | 0.9433 ± 0.0072 | 0.0111 ± 0.0039 | 0.9893 ± 0.0112 | 0.9659 ± 0.0036 | 0.9920 ± 0.0006 | 0.9968 ± 0.0003 |
| CSE-CIC-IDS2018 | 30 | flow + neighbourhood + host | 10 | 0.9714 ± 0.0024 | 0.9511 ± 0.0059 | 0.0116 ± 0.0052 | 0.9858 ± 0.0191 | 0.9675 ± 0.0088 | 0.9921 ± 0.0009 | 0.9967 ± 0.0003 |

| source | gap_min | representation | mean diff vs baseline | seeds better | seeds | Wilcoxon p |
|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | 1 | flow + host | -0.0004 | 6 | 10 | 1.0000 |
| CSE-CIC-IDS2018 | 1 | flow + neighbourhood + host | 0.0243 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | 5 | flow + host | 0.0486 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | 5 | flow + neighbourhood + host | 0.0499 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | 30 | flow + host | 0.0265 | 10 | 10 | 0.0020 |
| CSE-CIC-IDS2018 | 30 | flow + neighbourhood + host | 0.0280 | 10 | 10 | 0.0020 |
| ToN-IoT | 1 | flow + host | -0.1020 | 1 | 10 | 0.0137 |
| ToN-IoT | 1 | flow + neighbourhood + host | -0.0608 | 2 | 10 | 0.1055 |
| ToN-IoT | 5 | flow + host | -0.0881 | 2 | 10 | 0.0195 |
| ToN-IoT | 5 | flow + neighbourhood + host | -0.0543 | 5 | 10 | 0.2754 |
| ToN-IoT | 30 | flow + host | -0.0811 | 3 | 10 | 0.0371 |
| ToN-IoT | 30 | flow + neighbourhood + host | -0.0734 | 3 | 10 | 0.0840 |

Episodes per family by gap (minutes):

| source | family | 1 | 5 | 30 |
|---|---|---|---|---|
| CSE-CIC-IDS2018 | Bot | 5.0000 | 1.0000 | 1.0000 |
| CSE-CIC-IDS2018 | Brute_Force_-Web | 160.0000 | 23.0000 | 1.0000 |
| CSE-CIC-IDS2018 | Brute_Force_-XSS | 79.0000 | 39.0000 | 1.0000 |
| CSE-CIC-IDS2018 | DDOS_attack-HOIC | 1.0000 | 1.0000 | 1.0000 |
| CSE-CIC-IDS2018 | DDOS_attack-LOIC-UDP | 1.0000 | 1.0000 | 1.0000 |
| CSE-CIC-IDS2018 | DDoS_attacks-LOIC-HTTP | 8.0000 | 4.0000 | 2.0000 |
| CSE-CIC-IDS2018 | DoS_attacks-GoldenEye | 4.0000 | 4.0000 | 1.0000 |
| CSE-CIC-IDS2018 | DoS_attacks-Hulk | 1.0000 | 1.0000 | 1.0000 |
| CSE-CIC-IDS2018 | DoS_attacks-SlowHTTPTest | 1.0000 | 1.0000 | 1.0000 |
| CSE-CIC-IDS2018 | DoS_attacks-Slowloris | 1.0000 | 1.0000 | 1.0000 |
| CSE-CIC-IDS2018 | FTP-BruteForce | 1.0000 | 1.0000 | 1.0000 |
| CSE-CIC-IDS2018 | Infilteration | 314.0000 | 12.0000 | 2.0000 |
| CSE-CIC-IDS2018 | SQL_Injection | 69.0000 | 16.0000 | 2.0000 |
| CSE-CIC-IDS2018 | SSH-Bruteforce | 1.0000 | 1.0000 | 1.0000 |
| ToN-IoT | Backdoor | 228.0000 | 4.0000 | 1.0000 |
| ToN-IoT | DDoS | 50.0000 | 2.0000 | 2.0000 |
| ToN-IoT | DoS | 3.0000 | 1.0000 | 1.0000 |
| ToN-IoT | injection | 21.0000 | 3.0000 | 2.0000 |
| ToN-IoT | mitm | 27.0000 | 4.0000 | 1.0000 |
| ToN-IoT | password | 37.0000 | 2.0000 | 1.0000 |
| ToN-IoT | ransomware | 22.0000 | 3.0000 | 2.0000 |
| ToN-IoT | scanning | 55.0000 | 23.0000 | 3.0000 |
| ToN-IoT | xss | 6.0000 | 1.0000 | 1.0000 |

### 5.10 External validation — UNSW-NB15 (original)

| dataset | rows | attack_share | distinct_vectors | irreducible_error | attack_flows_in_ambiguous_vectors | duplicate_rows |
|---|---|---|---|---|---|---|
| unsw | 2,540,047 | 0.1265 | 2,016,173 | 0.0003 | 0.0028 | 0.2062 |

| attack | flows | distinct_vectors | share_identical_to_benign |
|---|---|---|---|
| Generic | 215,481 | 18,830 | 0.0000 |
| Exploits | 44,525 | 27,434 | 0.0000 |
| Fuzzers | 24,246 | 20,960 | 0.0372 |
| DoS | 16,353 | 5,500 | 0.0000 |
| Reconnaissance | 13,987 | 9,991 | 0.0000 |
| Analysis | 2,677 | 2,032 | 0.0000 |
| Backdoor | 2,329 | 1,880 | 0.0000 |
| Shellcode | 1,511 | 1,456 | 0.0033 |
| Worms | 174 | 171 | 0.0000 |

Single-feature shortcuts (top 10):

| feature | family_balanced_accuracy | binary_balanced_accuracy |
|---|---|---|
| sttl | 0.2404 | 0.9894 |
| ct_state_ttl | 0.2626 | 0.9868 |
| dttl | 0.2521 | 0.9385 |
| sbytes | 0.4095 | 0.9201 |
| smeansz | 0.3949 | 0.8966 |
| dbytes | 0.2852 | 0.8943 |
| dload | 0.2631 | 0.8919 |
| sload | 0.3410 | 0.8882 |
| dintpkt | 0.2494 | 0.8816 |
| dmeansz | 0.2719 | 0.8786 |

| representation | split | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC | test rows duplicated in train |
|---|---|---|---|---|---|---|---|---|---|---|
| behaviour | random rows | 5 | 0.9931 ± 0.0018 | 0.9298 ± 0.0018 | 0.0095 ± 0.0013 | 0.9704 ± 0.0049 | 0.9705 ± 0.0017 | 0.9978 ± 0.0003 | 0.9991 ± 0.0001 | 0.2847 |
| timestamp only | random rows | 5 | 0.0896 ± 0.0044 | 0.0125 ± 0.0060 | 0.0088 ± 0.0013 | 0.2526 ± 0.0070 | 0.3803 ± 0.0047 | 0.3339 ± 0.0040 | 0.7987 ± 0.0017 | 0.2847 |
| behaviour | 10-min time blocks | 5 | 0.9883 ± 0.0075 | 0.9300 ± 0.0122 | 0.0077 ± 0.0026 | 0.9730 ± 0.0149 | 0.9691 ± 0.0099 | 0.9974 ± 0.0013 | 0.9990 ± 0.0006 | 0.0810 |
| timestamp only | 10-min time blocks | 5 | 0.1854 ± 0.2137 | 0.1854 ± 0.2137 | 0.0657 ± 0.0649 | 0.2027 ± 0.1190 | 0.3055 ± 0.1809 | 0.2554 ± 0.0526 | 0.7350 ± 0.0811 | 0.0810 |
| behaviour | attack episodes | 5 | 0.9698 ± 0.0131 | 0.9022 ± 0.0240 | 0.0076 ± 0.0016 | 0.9740 ± 0.0104 | 0.9600 ± 0.0061 | 0.9961 ± 0.0009 | 0.9980 ± 0.0011 | 0.2442 |
| timestamp only | attack episodes | 5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2207 ± 0.0555 | 0.3588 ± 0.0737 | 0.1608 ± 0.0488 | 0.5940 ± 0.1458 | 0.2442 |

Official training/testing partition (gradient boosting):

| split | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC | test rows duplicated in train |
|---|---|---|---|---|---|---|---|---|---|
| official partition | 5 | 0.9091 ± 0.0026 | 0.8060 ± 0.0235 | 0.0554 ± 0.0037 | 0.8777 ± 0.0089 | 0.9257 ± 0.0037 | 0.9907 ± 0.0002 | 0.9840 ± 0.0004 | 0.1037 |

Context features and E-GraphSAGE (attack-episode split, 10 seeds):

| model | representation | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| HistGB | flow | 10 | 0.9734 ± 0.0157 | 0.8739 ± 0.0325 | 0.0083 ± 0.0027 | 0.9511 ± 0.0222 | 0.9585 ± 0.0048 | 0.9951 ± 0.0012 | 0.9986 ± 0.0004 |
| HistGB | flow + neighbourhood | 10 | 0.9680 ± 0.0139 | 0.8775 ± 0.0258 | 0.0083 ± 0.0028 | 0.9449 ± 0.0266 | 0.9565 ± 0.0062 | 0.9951 ± 0.0013 | 0.9985 ± 0.0005 |
| HistGB | flow + host | 10 | 0.9716 ± 0.0176 | 0.8760 ± 0.0435 | 0.0082 ± 0.0024 | 0.9589 ± 0.0285 | 0.9545 ± 0.0090 | 0.9959 ± 0.0008 | 0.9990 ± 0.0003 |
| HistGB | flow + neighbourhood + host | 10 | 0.9739 ± 0.0092 | 0.8939 ± 0.0290 | 0.0080 ± 0.0019 | 0.9521 ± 0.0297 | 0.9576 ± 0.0042 | 0.9961 ± 0.0008 | 0.9989 ± 0.0003 |
| RandomForest | flow | 10 | 0.9669 ± 0.0248 | 0.7840 ± 0.0824 | 0.0085 ± 0.0025 | 0.9369 ± 0.0188 | 0.9584 ± 0.0062 | 0.9909 ± 0.0049 | 0.9980 ± 0.0008 |
| RandomForest | flow + neighbourhood | 10 | 0.9596 ± 0.0245 | 0.8382 ± 0.0338 | 0.0086 ± 0.0025 | 0.9362 ± 0.0203 | 0.9563 ± 0.0069 | 0.9931 ± 0.0025 | 0.9980 ± 0.0008 |
| RandomForest | flow + host | 10 | 0.9544 ± 0.0367 | 0.8034 ± 0.0696 | 0.0086 ± 0.0026 | 0.9313 ± 0.0226 | 0.9499 ± 0.0141 | 0.9929 ± 0.0024 | 0.9985 ± 0.0006 |
| RandomForest | flow + neighbourhood + host | 10 | 0.9550 ± 0.0311 | 0.8282 ± 0.0564 | 0.0084 ± 0.0026 | 0.9355 ± 0.0220 | 0.9552 ± 0.0101 | 0.9933 ± 0.0023 | 0.9983 ± 0.0007 |
| MLP | flow | 10 | 0.9522 ± 0.0235 | 0.8224 ± 0.0366 | 0.0087 ± 0.0029 | 0.9279 ± 0.0195 | 0.9551 ± 0.0083 | 0.9919 ± 0.0023 | 0.9976 ± 0.0008 |
| MLP | flow + neighbourhood | 10 | 0.9592 ± 0.0243 | 0.8548 ± 0.0424 | 0.0082 ± 0.0029 | 0.9329 ± 0.0222 | 0.9535 ± 0.0057 | 0.9943 ± 0.0010 | 0.9981 ± 0.0005 |
| MLP | flow + host | 10 | 0.9625 ± 0.0263 | 0.8457 ± 0.0485 | 0.0082 ± 0.0025 | 0.9435 ± 0.0228 | 0.9532 ± 0.0082 | 0.9949 ± 0.0009 | 0.9987 ± 0.0004 |
| MLP | flow + neighbourhood + host | 10 | 0.9699 ± 0.0235 | 0.8830 ± 0.0367 | 0.0088 ± 0.0028 | 0.9550 ± 0.0217 | 0.9590 ± 0.0065 | 0.9953 ± 0.0016 | 0.9987 ± 0.0004 |
| E-GraphSAGE (ip) | flow | 10 | 0.6370 ± 0.3280 | 0.1016 ± 0.2084 | 0.0095 ± 0.0036 | 0.7898 ± 0.1923 | 0.8003 ± 0.2100 | 0.8703 ± 0.1055 | 0.9757 ± 0.0304 |
| E-GraphSAGE (ip_port) | flow | 10 | 0.9690 ± 0.0192 | 0.8695 ± 0.0507 | 0.0091 ± 0.0028 | 0.9411 ± 0.0239 | 0.9579 ± 0.0091 | 0.9924 ± 0.0077 | 0.9982 ± 0.0012 |
| E-GraphSAGE (random) | flow | 10 | 0.9489 ± 0.0234 | 0.7499 ± 0.2654 | 0.0091 ± 0.0034 | 0.9223 ± 0.0305 | 0.9500 ± 0.0119 | 0.9817 ± 0.0330 | 0.9970 ± 0.0021 |

| dataset | model | representation | mean diff vs baseline | seeds better | seeds | Wilcoxon p |
|---|---|---|---|---|---|---|
| unsw | HistGB | flow + host | -0.0018 | 4 | 10 | 0.9219 |
| unsw | HistGB | flow + neighbourhood | -0.0053 | 2 | 10 | 0.1602 |
| unsw | HistGB | flow + neighbourhood + host | 0.0005 | 4 | 10 | 0.9219 |
| unsw | MLP | flow + host | 0.0103 | 7 | 10 | 0.1055 |
| unsw | MLP | flow + neighbourhood | 0.0070 | 9 | 10 | 0.0195 |
| unsw | MLP | flow + neighbourhood + host | 0.0177 | 9 | 10 | 0.0059 |
| unsw | RandomForest | flow + host | -0.0125 | 5 | 10 | 0.4922 |
| unsw | RandomForest | flow + neighbourhood | -0.0073 | 0 | 10 | 0.0020 |
| unsw | RandomForest | flow + neighbourhood + host | -0.0119 | 1 | 10 | 0.0371 |
| unsw | E-GraphSAGE (ip) | – | -0.3364 | 0 | 10 | 0.0020 |
| unsw | E-GraphSAGE (ip_port) | – | -0.0044 | 4 | 10 | 0.4316 |
| unsw | E-GraphSAGE (random) | – | -0.0244 | 0 | 10 | 0.0020 |

### 5.11 External validation — CIC-IoT2023

| dataset | rows | attack_share | distinct_vectors | irreducible_error | attack_flows_in_ambiguous_vectors | duplicate_rows |
|---|---|---|---|---|---|---|
| ciciot | 46,686,579 | 0.9765 | 46,686,545 | 0.0000 | 0.0000 | 0.0000 |

Single-feature shortcuts (top 10):

| feature | family_balanced_accuracy | binary_balanced_accuracy |
|---|---|---|
| IAT | 0.6731 | 0.9363 |
| Number | 0.0587 | 0.8716 |
| Weight | 0.0587 | 0.8715 |
| rst_count | 0.1701 | 0.8683 |
| urg_count | 0.1309 | 0.8578 |
| Variance | 0.1260 | 0.8538 |
| flow_duration | 0.1464 | 0.8071 |
| Header_Length | 0.2209 | 0.8066 |
| HTTPS | 0.0294 | 0.7961 |
| Duration | 0.0946 | 0.7954 |

| representation | split | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC | test rows duplicated in train |
|---|---|---|---|---|---|---|---|---|---|---|
| behaviour | random rows | 5 | 0.9969 ± 0.0001 | 0.9951 ± 0.0002 | 0.0096 ± 0.0008 | 0.9992 ± 0.0001 | 0.9987 ± 0.0000 | 1.0000 ± 0.0000 | 0.9964 ± 0.0002 | 0.0000 |

Context features (random split, 10 seeds):

| model | representation | seeds | TPR@1%FPR | TPR@0.1%FPR | FPR at 1% thr. | Precision (w) | F1 (w) | PR-AUC (w) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| HistGB | flow | 10 | 0.9959 ± 0.0004 | 0.9940 ± 0.0009 | 0.0101 ± 0.0045 | 0.9988 ± 0.0003 | 0.9984 ± 0.0001 | 1.0000 ± 0.0000 | 0.9950 ± 0.0004 |
| HistGB | flow + neighbourhood | 10 | 0.9961 ± 0.0005 | 0.9939 ± 0.0009 | 0.0103 ± 0.0028 | 0.9986 ± 0.0002 | 0.9984 ± 0.0001 | 1.0000 ± 0.0000 | 0.9951 ± 0.0004 |
| RandomForest | flow | 10 | 0.9951 ± 0.0006 | 0.9887 ± 0.0027 | 0.0122 ± 0.0049 | 0.9985 ± 0.0003 | 0.9984 ± 0.0001 | 1.0000 ± 0.0000 | 0.9935 ± 0.0005 |
| RandomForest | flow + neighbourhood | 10 | 0.9942 ± 0.0011 | 0.9892 ± 0.0011 | 0.0110 ± 0.0046 | 0.9985 ± 0.0003 | 0.9982 ± 0.0001 | 1.0000 ± 0.0000 | 0.9923 ± 0.0005 |
| MLP | flow | 10 | 0.9910 ± 0.0008 | 0.9804 ± 0.0085 | 0.0109 ± 0.0039 | 0.9977 ± 0.0004 | 0.9969 ± 0.0001 | 1.0000 ± 0.0000 | 0.9793 ± 0.0010 |
| MLP | flow + neighbourhood | 10 | 0.9892 ± 0.0010 | 0.8863 ± 0.3114 | 0.0089 ± 0.0023 | 0.9975 ± 0.0004 | 0.9966 ± 0.0001 | 0.9999 ± 0.0000 | 0.9751 ± 0.0016 |

| dataset | model | representation | mean diff vs baseline | seeds better | seeds | Wilcoxon p |
|---|---|---|---|---|---|---|
| ciciot | HistGB | flow + neighbourhood | 0.0001 | 6 | 10 | 0.4316 |
| ciciot | MLP | flow + neighbourhood | -0.0018 | 0 | 10 | 0.0020 |
| ciciot | RandomForest | flow + neighbourhood | -0.0010 | 1 | 10 | 0.0039 |

### 5.12 Experiment wall-clock times (BigFlow pipeline)

Not a detection-latency measure; end-to-end time of each experiment stage on the hardware in §3.

| stage | start | wall-clock |
|---|---|---|
| label ambiguity (all flows) | 2026-10-09 13:58:39 | 0 days 00:12:18 |
| working sample + host features | 2026-10-09 14:10:57 | 0 days 00:31:54 |
| part 1 shortcuts | 2026-10-09 14:42:51 | 0 days 00:06:12 |
| parts 3-4 splits + context | 2026-10-09 14:49:03 | 0 days 01:09:23 |
| part 5 two-branch | 2026-10-09 15:58:26 | 0 days 00:24:41 |
| robustness A (RF, MLP) | 2026-10-09 16:23:07 | 0 days 06:05:49 |
| robustness B (gap) | 2026-10-09 22:28:56 | 0 days 01:05:00 |
| E-GraphSAGE | 2026-10-09 23:33:56 | 0 days 07:58:02 |
| figures and tables | 2026-10-10 07:31:58 | 0 days 00:00:11 |

## 6. Ablations (where to find them)

- Context feature sets (flow / + neighbourhood / + host / + both): §5.5, §5.6, §5.10, §5.11.
- Graph structure (E-GraphSAGE IP vs IP:port vs random-node control): §5.7, §5.10.
- Adaptive routing vs plain ensemble (family-gated detector vs expert mean): §5.8.
- Split strategy: §5.3, §5.4, §5.10. Episode gap: §5.9.
- Structural centralities (degree, PageRank, component size) were dropped after an early 3-seed run on a 46M-flow subset showed them unstable; that run is not part of this export.

## 7. Statistical testing

All model comparisons: paired two-sided Wilcoxon signed-rank test over seeds (each seed = one split + one model fit). With 10 seeds the smallest attainable p is 0.002; with 5 it is 0.0625, so 5-seed comparisons (split strategies) are reported descriptively only. Tables above give `seeds better` (wins) and p for TPR@1%FPR; the CSVs in `results/` also contain the PR-AUC tests.

## 8. Known limitations

- Chronological fallback for classes with < 3 episodes (20% of ToN-IoT, 33% of CIC-IDS2018 test flows): the episode split is leakage-reduced, not leakage-free.
- Only ToN-IoT and CSE-CIC-IDS2018 support a meaningful benign-vs-attack comparison within BigFlow (BoT-IoT is 99.7% attacks; UNSW-NB15 saturates).
- CIC-IoT2023 has no timestamps or addresses (partial protocol), and its continuous features make exact-match ambiguity uninformative.
- E-GraphSAGE is a re-implementation.
- Latency, throughput and detection/response time were not measured.

## 9. Folder contents

- `figures/` — all paper figures (PDF + PNG).
- `results/bigflow/` — BigFlow result CSVs (raw per-seed, summaries, significance).
- `results/external/` — UNSW-NB15 and CIC-IoT2023 result CSVs and class counts.
- `results/paper_tables/` — the LaTeX tables of the paper and their CSVs.
- `logs/` — output logs of the final runs (see `logs/README.md`).
