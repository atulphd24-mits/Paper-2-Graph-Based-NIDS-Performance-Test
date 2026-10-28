"""
Compare subset (i) coverage on the episode-stratified working sample with the previous full
working sample (simple 20k cap) and the 60k subsample. Smoke checks at the main protocol
(1-min gap, 5-min embargo), seeds 0-4. Writes outputs/paper2_final/reports/smoke_strat_vs_full.{csv,md}.

Usage: python scripts/smoke_compare_strat.py
"""
from pathlib import Path

import pandas as pd

from smoke_compare import md

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/paper2_final/reports"
SAMPLES = (("60k", "60k subsample"), ("full", "full working sample (20k cap)"),
           ("strat", "episode-stratified sample"))


def main():
    cov, fam = [], []
    for d in ("bigflow", "unsw"):
        for key, label in SAMPLES:
            tag = f"_{d}_{key}_s5"
            if not (R / f"smoke_subsets{tag}.csv").exists():
                continue
            s = pd.read_csv(R / f"smoke_subsets{tag}.csv")
            a = pd.read_csv(R / f"smoke_family_assignment{tag}.csv")
            a = a[a.family != "Benign"].assign(sample=label)
            fam.append(a)
            n_fam = a[a.in_cross_subset].groupby(["source", "seed"]).size()
            for (src, seed), g in s[s.subset == "cross-episode"].groupby(["source", "seed"]):
                r = g.iloc[0]
                cov.append({"source": src, "sample": label, "seed": seed,
                            "share_of_test_attacks": r.share_of_test_attacks,
                            "attack_flows": r.test_attack_flows, "test_episodes": r.test_episodes,
                            "families": int(n_fam.get((src, seed), 0))})
    cov, fam = pd.DataFrame(cov), pd.concat(fam, ignore_index=True)
    cov.to_csv(R / "smoke_strat_vs_full_seeds.csv", index=False)

    summary = (cov.groupby(["source", "sample"], sort=False)
               .agg(share_mean=("share_of_test_attacks", "mean"), attack_flows_mean=("attack_flows", "mean"),
                    test_episodes_mean=("test_episodes", "mean"), test_episodes_min=("test_episodes", "min"),
                    families_mean=("families", "mean"), families_min=("families", "min"),
                    families_max=("families", "max"))
               .reset_index())
    summary.to_csv(R / "smoke_strat_vs_full.csv", index=False)

    per_family = (fam.groupby(["source", "family", "sample"], sort=False)
                  .agg(episodes_in_sample=("episodes_in_sample", "first"),
                       test_episodes_mean=("test_episodes", "mean"),
                       train_flows_mean=("train", "mean"),
                       seeds_in_i=("in_cross_subset", "sum"))
                  .reset_index())
    wide = per_family.pivot_table(index=["source", "family"], columns="sample",
                                  values=["episodes_in_sample", "test_episodes_mean", "seeds_in_i"],
                                  aggfunc="first")
    wide.columns = [f"{v} ({s})" for v, s in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(R / "smoke_strat_vs_full_families.csv", index=False)

    order = [f"{v} ({s})" for v in ("episodes_in_sample", "test_episodes_mean", "seeds_in_i")
             for _, s in SAMPLES if f"{v} ({s})" in wide.columns]
    lines = ["# Subset (i) coverage: episode-stratified vs previous samples\n",
             "Smoke check, main protocol (1-min gap, 5-min embargo), seeds 0-4. Subset (i) = attack flows "
             "of families with >= 3 test episodes and >= 500 training flows.\n",
             "## Per source (mean over 5 seeds; min/max where shown)\n", md(summary, 2),
             "\n## Per family\n",
             "`seeds_in_i`: number of the 5 seeds in which the family enters subset (i).\n",
             md(wide[["source", "family"] + order].fillna("-"), 1)]
    (R / "smoke_strat_vs_full.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
