# Paper-2-Graph-Based-Performance-Evaluation
Goal: find out how much of the reported performance of machine-learning intrusion detectors survives a realistic, leakage-free evaluation, measuring both attacks detected and false alarms on new traffic.

## Objective

Intrusion detection models often look excellent in research papers but perform much worse
when they meet new traffic. This project measures **how much of that reported performance is real**.

We test machine-learning and graph-based network intrusion detectors on **BigFlow-NIDS v2**
(66.9 million network flows merged from four public datasets), and we evaluate them the way they
would be used in a real Security Operations Centre (SOC):

- **Train on past attacks, test on new ones.** Whole attack sessions ("episodes") go to either
  training or testing, never both, with a time gap between them so no information leaks across.
- **Use only information available at detection time.** Host features look only at past traffic;
  neighbourhood features and graph models never see test data during training.
- **Fix the false-alarm budget in advance.** Alert thresholds are set on validation data for a
  1% false-positive rate, and we report the false-alarm rate the detector actually reaches on test.

### Questions we answer

1. Do high detection rates reflect the detector, or just which attacks happen to be in the test set?
2. Does a false-alarm target set in advance still hold on new traffic?
3. Do context features (host behaviour, neighbouring flows) and graph models (E-GraphSAGE) really
   improve detection once leakage is removed?
4. How large are the dataset's own flaws (duplicate flows labelled both attack and benign,
   traffic whose source dataset can be identified)?

### Why it matters

A SOC can only handle a limited number of alerts. A detector is useful only if it finds attacks
**and** keeps its false alarms within budget on traffic it has never seen. This project gives an
evaluation protocol and evidence for checking both.

This work addresses research objective 1 of the PhD: *to design an AI-based intelligent model to
enhance attack detection accuracy and reduce false alarms in Security Information and Event
Management (SIEM) systems.*