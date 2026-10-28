"""
Finding 2 - label ambiguity on the complete BigFlow data.

For every flow, the behaviour vector (all columns except identifiers,
timestamps and targets) is hashed. Per source dataset we report:

* share of each attack family whose exact vector also occurs as Benign
* irreducible binary error: sum over vectors of min(#benign, #attack) / N,
  a lower bound on the error of ANY per-flow classifier using these features

Usage:
    python scripts/label_ambiguity.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bigflow import bigflow_dir, behaviour_columns, list_files, read_file  # noqa: E402

DATA_DIR = bigflow_dir()
OUTPUT_DIR = PROJECT_ROOT / "outputs/reports/leakage_study"


def main():
    start = time.time()
    parts = []
    columns = None

    for path in list_files(DATA_DIR):
        df = read_file(path)
        columns = columns or behaviour_columns(df.columns)
        parts.append(
            pd.DataFrame(
                {
                    "hash": pd.util.hash_pandas_object(df[columns], index=False).to_numpy(),
                    "source": df["source"].astype("category"),
                    "attack": df["attack"].astype("category"),
                }
            )
        )
        print(f"[{time.time() - start:5.0f}s] hashed {Path(path).name}", flush=True)

    flows = pd.concat(parts, ignore_index=True)
    flows["source"] = flows["source"].astype("category")
    flows["attack"] = flows["attack"].astype("category")
    flows["is_attack"] = flows["attack"] != "Benign"
    print(f"Behaviour columns: {len(columns)} | flows: {len(flows):,}")

    family_rows, source_rows = [], []
    for source, group in flows.groupby("source", observed=True):
        benign_hashes = np.unique(group.loc[~group["is_attack"], "hash"].to_numpy())

        for attack, fam in group[group["is_attack"]].groupby("attack", observed=True):
            family_rows.append({
                "source": source,
                "attack": attack,
                "flows": len(fam),
                "distinct_vectors": fam["hash"].nunique(),
                "share_identical_to_benign": np.isin(fam["hash"].to_numpy(), benign_hashes).mean(),
            })

        per_vector = group.groupby(["hash", "is_attack"]).size().unstack(fill_value=0)
        if True not in per_vector:
            per_vector[True] = 0
        if False not in per_vector:
            per_vector[False] = 0
        source_rows.append({
            "source": source,
            "flows": len(group),
            "attack_share": group["is_attack"].mean(),
            "distinct_vectors": len(per_vector),
            "irreducible_error": np.minimum(per_vector[True], per_vector[False]).sum() / len(group),
            "attack_flows_in_ambiguous_vectors": (
                per_vector.loc[per_vector[False] > 0, True].sum() / max(group["is_attack"].sum(), 1)
            ),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    families = pd.DataFrame(family_rows).sort_values(["source", "flows"], ascending=[True, False])
    sources = pd.DataFrame(source_rows)
    families.to_csv(OUTPUT_DIR / "label_ambiguity_by_family.csv", index=False)
    sources.to_csv(OUTPUT_DIR / "label_ambiguity_by_source.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nPER SOURCE\n", sources.round(4).to_string(index=False))
    print("\nPER FAMILY\n", families.round(4).to_string(index=False))
    print(f"\nDone in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
