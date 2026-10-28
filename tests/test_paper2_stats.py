import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper2_stats import descriptive, egs_frame, holm_f7, run_tests  # noqa: E402


def rows(conds, seeds, subsets, offset):
    out = []
    for c in conds:
        for s in seeds:
            for sub in subsets:
                out.append({"source": "S", "cond": c, "seed": s, "subset": sub,
                            "TPR@FPR1pct": 0.5 + offset[c] + 0.01 * s, "PR_AUC_w": 0.6 + offset[c],
                            "TPR@FPR0.1pct": 0.3, "testFPR@1pct": 0.01, "test_episodes": 4})
    return out


class StatsTests(unittest.TestCase):
    def test_subset_i_tested_only_when_every_seed_has_it(self):
        d = pd.DataFrame(rows(["base", "x"], range(10), ["all"], {"base": 0, "x": 0.1})
                         + rows(["base", "x"], range(9), ["cross-episode"], {"base": 0, "x": 0.1}))
        t = run_tests("F", d, "cond", "base", ["source"])
        self.assertEqual(set(t.subset), {"(iii) all"})  # seed 9 lacks (i): not tested on (i)
        d2 = pd.concat([d, pd.DataFrame(rows(["base", "x"], [9], ["cross-episode"], {"base": 0, "x": 0.1}))])
        t2 = run_tests("F", d2, "cond", "base", ["source"])
        self.assertEqual(set(t2.subset), {"(iii) all", "(i) cross-episode"})
        r = t2[(t2.metric == "TPR@FPR1pct") & (t2.subset == "(i) cross-episode")].iloc[0]
        self.assertAlmostEqual(r.mean_diff, 0.1)
        self.assertEqual(r.seeds, 10)

    def test_descriptive_mean_and_ci(self):
        d = pd.DataFrame(rows(["x"], range(10), ["all"], {"x": 0}))
        out = descriptive(d, ["source", "cond"]).iloc[0]
        self.assertAlmostEqual(out["TPR@FPR1pct_mean"], 0.545)
        self.assertLessEqual(out["TPR@FPR1pct_ci_low"], 0.545)
        self.assertGreaterEqual(out["TPR@FPR1pct_ci_high"], 0.545)

    def test_baseline_shared_with_both_modes(self):
        eg = pd.DataFrame({"source": "S", "seed": 0, "protocol": "p", "subset": "all",
                           "model": ["HistGB (flow, no graph)", "E-GraphSAGE (ip)", "E-GraphSAGE (ip)"],
                           "mode": ["-", "inductive", "transductive"]})
        out = egs_frame(eg)
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted(out[out.model.str.startswith("HistGB")]["mode"]), ["inductive", "transductive"])

    def test_f7_holm_over_endpoints(self):
        t = pd.DataFrame({"family": ["F7 x"] * 3 + ["F6 y"], "source": "S", "protocol": "p", "metric": "m",
                          "subset": "s", "model": ["a", "b", "c", "a"], "p_raw": [0.01, 0.04, 0.03, 0.01],
                          "p_holm": [0.01, 0.04, 0.03, 0.02], "family_size": [1, 1, 1, 3]})
        out = holm_f7(t.copy())
        self.assertEqual(out.family_size.tolist()[:3], [3, 3, 3])
        np.testing.assert_allclose(out.p_holm[:3], [0.03, 0.06, 0.06])
        self.assertEqual(out.p_holm.iloc[3], 0.02)  # other families untouched


if __name__ == "__main__":
    unittest.main()
