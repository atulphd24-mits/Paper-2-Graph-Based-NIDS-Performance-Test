"""
Expected disk size of the per-flow prediction files for the frozen run plan (option b-i, which
contains b-ii). Rows per file = validation + test rows per source (smoke check, stratified sample,
mean over seeds 0-4); bytes per row from the UNSW-NB15 check files (zstd parquet): measured
~11 B/row metadata and 1.7-3.2 B/row per float32 score column. The upper bound assumes
uncompressed scores (4 B) and 14 B/row metadata.

Writes outputs/paper2_final/reports/prediction_disk_estimate.md.
"""
from pathlib import Path

import pandas as pd

from smoke_compare import md

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/paper2_final/reports"

ROWS = {"ToN-IoT": 74_746, "CSE-CIC-IDS2018": 91_615, "UNSW-NB15": 58_742, "CIC-IoT2023": 191_942}
BIGFLOW = ["ToN-IoT", "CSE-CIC-IDS2018"]
# (part, sources, files per seed per source, score columns per file, seeds, in b-ii)
PLAN = [
    ("splits", BIGFLOW, 3, 4, 10, True),
    ("context HistGB main", BIGFLOW, 1, 4, 30, True),
    ("context RF + MLP main", BIGFLOW, 1, 8, 10, True),
    ("context HistGB gap 5 / chronological / E=0", BIGFLOW, 3, 4, 10, True),
    ("gated (reused + split validation)", BIGFLOW, 2, 5, 30, True),
    ("E-GraphSAGE inductive main (+ baseline)", BIGFLOW, 1, 4, 30, True),
    ("E-GraphSAGE inductive random (+ baseline)", BIGFLOW, 1, 4, 10, True),
    ("E-GraphSAGE inductive gap 5 (+ baseline)", BIGFLOW, 1, 4, 10, False),
    ("E-GraphSAGE transductive main / random", BIGFLOW, 2, 3, 10, True),
    ("UNSW-NB15 splits", ["UNSW-NB15"], 3, 4, 10, True),
    ("UNSW-NB15 context", ["UNSW-NB15"], 1, 12, 10, True),
    ("UNSW-NB15 E-GraphSAGE", ["UNSW-NB15"], 2, 6, 10, True),
    ("CIC-IoT2023 splits", ["CIC-IoT2023"], 1, 3, 10, True),
    ("CIC-IoT2023 context", ["CIC-IoT2023"], 1, 6, 10, True),
]
MEASURED = (11.0, 2.5)  # B/row metadata, B/row/score column
UPPER = (14.0, 4.0)


def main():
    out = []
    for part, sources, files, cols, seeds, b2 in PLAN:
        rows = sum(ROWS[s] for s in sources) * files * seeds
        out.append({"part": part, "files": len(sources) * files * seeds, "rows_M": rows / 1e6,
                    "score_columns": cols, "in_b_ii": b2,
                    "expected_GB": rows * (MEASURED[0] + cols * MEASURED[1]) / 1e9,
                    "upper_GB": rows * (UPPER[0] + cols * UPPER[1]) / 1e9})
    t = pd.DataFrame(out)
    tot = pd.DataFrame([{"option": o, "files": int(s["files"].sum()), "expected_GB": s["expected_GB"].sum(),
                         "upper_GB": s["upper_GB"].sum()}
                        for o, s in (("(b-i)", t), ("(b-ii)", t[t.in_b_ii]))])
    lines = ["# Expected disk size of per-flow predictions\n", __doc__.strip() + "\n", md(t, 3),
             "\n## Totals\n", md(tot, 2)]
    (R / "prediction_disk_estimate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
