import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper2_final  # noqa: E402
from paper2_final import test_subsets  # noqa: E402


class TestSubsetTests(unittest.TestCase):
    def setUp(self):
        self._min_train = paper2_final.MIN_TRAIN_FLOWS
        paper2_final.MIN_TRAIN_FLOWS = 1  # tiny synthetic data

    def tearDown(self):
        paper2_final.MIN_TRAIN_FLOWS = self._min_train

    def test_family_with_too_few_training_flows_is_excluded(self):
        paper2_final.MIN_TRAIN_FLOWS = 5
        attack = np.array(["a"] * 8 + ["Benign"] * 2)
        label = (attack != "Benign").astype(int)
        split = np.array(["train", "train"] + ["test"] * 8, dtype=object)
        mode = np.full(10, "episode", dtype=object)
        episode = np.array(["a|0", "a|0", "a|1", "a|2", "a|3", "a|4", "a|5", "a|6", None, None], dtype=object)
        subsets, info = test_subsets(split, episode, mode, attack, label)
        self.assertEqual(info["cross_families"], [])
        self.assertEqual(info["low_training_flows"], ["a"])

    def test_three_subsets(self):
        # family a: 4 test episodes (eligible); b: 2 test episodes (dropped); c: fallback
        attack = np.array(["a"] * 8 + ["b"] * 4 + ["c"] * 4 + ["Benign"] * 6)
        label = (attack != "Benign").astype(int)
        split = np.array(["test"] * 22, dtype=object)
        split[[0, 1]] = "train"
        mode = np.where(attack == "c", "chronological", "episode").astype(object)
        episode = np.array(["a|0", "a|0", "a|1", "a|2", "a|3", "a|4", "a|4", "a|4",
                            "b|0", "b|0", "b|1", "b|1", "c|0", "c|0", "c|0", "c|0"] + [None] * 6,
                           dtype=object)
        subsets, info = test_subsets(split, episode, mode, attack, label)

        cross, fallback, every = subsets["cross-episode"], subsets["fallback"], subsets["all"]
        benign = attack == "Benign"
        # benign test flows are in every subset (same FPR denominator)
        self.assertTrue((cross[benign] & fallback[benign] & every[benign]).all())
        self.assertTrue(cross[(attack == "a") & (split == "test")].all())
        self.assertFalse(cross[attack == "b"].any())
        self.assertFalse(cross[attack == "c"].any())
        self.assertTrue(fallback[attack == "c"].all())
        self.assertFalse(fallback[attack == "a"].any())
        self.assertTrue((every == (split == "test")).all())
        self.assertEqual(info["cross_families"], ["a"])
        self.assertEqual(info["dropped_from_cross"], ["b", "c"])

    def test_random_split_has_only_all(self):
        n = 10
        subsets, _ = test_subsets(np.array(["test"] * n, dtype=object), np.full(n, None, object),
                                  np.full(n, "random", object), np.array(["x"] * 5 + ["Benign"] * 5),
                                  np.array([1] * 5 + [0] * 5))
        self.assertEqual(list(subsets), ["all"])


if __name__ == "__main__":
    unittest.main()
