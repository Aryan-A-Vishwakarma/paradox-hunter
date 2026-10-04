import unittest

import numpy as np
import pandas as pd

from paradox_hunter import datasets, scan
from paradox_hunter.engine import _mantel_haenszel, benjamini_hochberg, binarize_target


class TestStatistics(unittest.TestCase):
    def test_mantel_haenszel_matches_published_berkeley_value(self):
        # Berkeley: (male admitted, male rejected, female admitted, female rejected) per dept
        t = [(512, 313, 89, 19), (353, 207, 17, 8), (120, 205, 202, 391),
             (138, 279, 131, 244), (53, 138, 94, 299), (22, 351, 24, 317)]
        or_mh, p = _mantel_haenszel(t)
        self.assertAlmostEqual(or_mh, 0.905, places=2)   # literature: ~0.90
        self.assertGreater(p, 0.1)                       # not significant

    def test_benjamini_hochberg_monotone_and_bounded(self):
        p = np.array([0.001, 0.01, 0.03, 0.04, 0.5])
        q = benjamini_hochberg(p)
        self.assertTrue(np.all(q >= p))
        self.assertTrue(np.all(q <= 1))
        self.assertTrue(np.all(np.diff(q[np.argsort(p)]) >= 0))

    def test_binarize_target(self):
        self.assertEqual(binarize_target(pd.Series(["yes", "no", "yes"]), positive="yes").tolist(), [1, 0, 1])
        self.assertEqual(binarize_target(pd.Series([1, 2, 3, 4])).tolist(), [0, 0, 1, 1])


class TestDetection(unittest.TestCase):
    def test_berkeley_flagged(self):
        fs, _ = scan(datasets.berkeley_admissions(), "admitted")
        self.assertEqual(len(fs), 1)
        f = fs[0]
        self.assertEqual(f.stratifier, "department")
        self.assertGreater(f.crude_or, 1.5)
        self.assertLess(f.adj_or, 1.0)

    def test_kidney_stones_full_reversal(self):
        fs, _ = scan(datasets.kidney_stones(), "success", min_evidence="exploratory")
        self.assertEqual(len(fs), 1)
        self.assertTrue(fs[0].full_reversal)
        self.assertEqual(fs[0].consistency, 1.0)

    def test_planted_paradox_found_and_decoys_ignored(self):
        fs, s = scan(datasets.synthetic_hospital(), "died")
        self.assertEqual([(f.exposure, f.stratifier) for f in fs], [("hospital = B", "severity")])
        self.assertEqual(fs[0].evidence, "confirmed")
        self.assertGreater(s["tests_run"], 20)

    def test_no_false_positives_on_pure_noise(self):
        """Across many random datasets, nothing should be reported at 'suggestive' or above."""
        false_alarms = 0
        for seed in range(15):
            rng = np.random.default_rng(seed)
            n = 3000
            df = pd.DataFrame({
                "a": rng.choice(["x", "y"], n), "b": rng.choice(["p", "q", "r"], n),
                "c": rng.integers(0, 100, n), "d": rng.choice(["u", "v"], n),
                "e": rng.normal(size=n), "target": rng.integers(0, 2, n),
            })
            fs, _ = scan(df, "target")
            false_alarms += len(fs)
        self.assertEqual(false_alarms, 0)

    def test_consistent_effect_is_not_a_paradox(self):
        """A real, consistent effect (no confounding) must not be flagged."""
        rng = np.random.default_rng(1)
        n = 5000
        x = rng.integers(0, 2, n)
        z = rng.choice(["s1", "s2", "s3"], n)
        y = (rng.random(n) < 0.2 + 0.2 * x).astype(int)
        fs, _ = scan(pd.DataFrame({"x": x, "z": z, "y": y}), "y", min_evidence="exploratory")
        self.assertEqual(fs, [])

    def test_missing_target_raises(self):
        with self.assertRaises(KeyError):
            scan(datasets.kidney_stones(), "nope")

    def test_handles_missing_values_and_id_columns(self):
        df = datasets.synthetic_hospital().copy()
        df["patient_id"] = np.arange(len(df))
        df.loc[df.sample(frac=0.1, random_state=0).index, "severity"] = np.nan
        fs, _ = scan(df, "died")
        self.assertTrue(any(f.stratifier == "severity" for f in fs))
        self.assertFalse(any("patient_id" in (f.exposure, f.stratifier) for f in fs))



class TestViz(unittest.TestCase):
    def test_overall_rates_reproduce_headline_numbers(self):
        from paradox_hunter.viz import overall_rates, strata_long, strata_table
        f = scan(datasets.berkeley_admissions(), "admitted")[0][0]
        ov = overall_rates(f).set_index("group")["rate"]
        self.assertAlmostEqual(ov.iloc[0], 1198 / 2691, places=4)   # men admitted overall
        self.assertAlmostEqual(ov.iloc[1], 557 / 1835, places=4)    # women admitted overall
        self.assertEqual(len(strata_long(f)), 12)                   # 6 departments x 2 groups
        self.assertIn("opposes_headline", strata_table(f).columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
