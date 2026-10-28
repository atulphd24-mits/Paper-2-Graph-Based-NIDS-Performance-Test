"""
Compare the smoke check on each source's full working sample with the 60k-flow subsample
(seed 0, 1-min gap, 5-min embargo). Writes outputs/paper2_final/reports/smoke_full_vs_60k.{csv,md}.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/paper2_final/reports"
P = ROOT / "data/processed"


def md(df, digits=3):
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    bounds = pd.concat([pd.read_parquet(P / "episode_bounds_bigflow.parquet"),
                        pd.read_parquet(P / "episode_bounds_unsw.parquet")])
    full_data = bounds[bounds.gap_min == 1].groupby(["source", "attack"]).size().rename("episodes_full_data")

    fam, cov = [], []
    for d in ("bigflow", "unsw"):
        for tag, label in ((f"_{d}", "60k subsample"), (f"_{d}_full", "full working sample")):
            a = pd.read_csv(R / f"smoke_family_assignment{tag}.csv").assign(sample=label)
            s = pd.read_csv(R / f"smoke_subsets{tag}.csv").assign(sample=label)
            fam.append(a)
            cross_fams = a[a.in_cross_subset].groupby("source")["family"].apply(lambda x: ", ".join(sorted(x)))
            for r in s[s.subset == "cross-episode"].itertuples():
                cov.append({"source": r.source, "sample": label,
                            "subset_i_share_of_test_attacks": r.share_of_test_attacks,
                            "subset_i_attack_flows": r.test_attack_flows, "subset_i_episodes": r.test_episodes,
                            "families_in_i": int(a[(a.source == r.source) & a.in_cross_subset].shape[0]),
                            "which": cross_fams.get(r.source, "")})
    fam = pd.concat(fam, ignore_index=True)
    fam = fam[fam.family != "Benign"]
    wide = fam.pivot_table(index=["source", "family"], columns="sample",
                           values=["episodes_in_sample", "test_episodes", "train", "test"], aggfunc="first")
    wide.columns = [f"{v} ({s})" for v, s in wide.columns]
    wide = wide.join(full_data.rename_axis(["source", "family"]), how="left").reset_index()
    inc = fam.pivot_table(index=["source", "family"], columns="sample", values="in_cross_subset", aggfunc="first")
    inc.columns = [f"in (i) ({s})" for s in inc.columns]
    wide = wide.merge(inc.reset_index(), on=["source", "family"])
    wide.to_csv(R / "smoke_full_vs_60k.csv", index=False)
    cov = pd.DataFrame(cov)
    cov.to_csv(R / "smoke_full_vs_60k_coverage.csv", index=False)

    cols = ["source", "family", "episodes_full_data", "episodes_in_sample (full working sample)",
            "episodes_in_sample (60k subsample)", "test_episodes (full working sample)",
            "test_episodes (60k subsample)", "in (i) (full working sample)", "in (i) (60k subsample)"]
    lines = ["# Smoke check: full working sample vs 60k subsample (seed 0, 1-min gap, 5-min embargo)\n",
             "## Subset (i) coverage\n", md(cov), "\n## Episodes per family\n",
             "`episodes_full_data`: all flows of the class in the complete dataset; `episodes_in_sample`: "
             "episodes that contain at least one sampled flow.\n", md(wide[cols].fillna("-"))]
    (R / "smoke_full_vs_60k.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
