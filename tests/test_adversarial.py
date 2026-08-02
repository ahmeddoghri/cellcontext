import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cellcontext
from adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from cellcontext_v2 import run, run_index_baseline
from eval_v2 import summarize


class AdversarialTest(unittest.TestCase):
    def test_holdout_disjoint_from_tuning(self):
        self.assertTrue(set(TUNING_SEEDS).isdisjoint(HOLDOUT_SEEDS))

    def test_original_benchmark_still_reproduces_exactly(self):
        result = cellcontext.run()
        self.assertEqual(result["global_mae"], 0.2699)
        self.assertEqual(result["context_mae"], 0.1813)
        self.assertEqual(result["mae_reduction_pct"], 32.8)

    def test_v2_index_baseline_matches_original_module_on_unshuffled_data(self):
        result = run_index_baseline(seed=7, shuffle_identity=False)
        self.assertEqual(result, cellcontext.run(seed=7))

    def test_original_bug_index_proximity_fails_when_index_is_not_identity(self):
        """cellcontext.py's nearest-context lookup uses literal array index
        proximity (min(abs(cc - c))). It only wins because the synthetic
        generator makes the interaction term a linear function of context
        index. Decouple index from the true latent driver (shuffle_identity)
        and the method should collapse below the naive global-average
        baseline on average."""
        reductions = [run_index_baseline(seed, shuffle_identity=True)["mae_reduction_pct"] for seed in TUNING_SEEDS]
        mean_reduction = sum(reductions) / len(reductions)
        self.assertLess(mean_reduction, 0)

    def test_v2_fix_generalizes_on_tuning_seeds(self):
        result = summarize(TUNING_SEEDS)
        self.assertGreater(result["data_mean_reduction_pct"], result["index_mean_reduction_pct"])
        self.assertGreater(result["data_mean_reduction_pct"], 20)

    def test_v2_fix_generalizes_on_frozen_holdout_seeds(self):
        result = summarize(HOLDOUT_SEEDS)
        self.assertGreater(result["data_mean_reduction_pct"], result["index_mean_reduction_pct"])
        self.assertGreater(result["data_mean_reduction_pct"], 20)

    def test_v2_does_not_regress_the_original_published_dataset(self):
        result = run(seed=7, shuffle_identity=False)
        self.assertGreaterEqual(result["mae_reduction_pct"], 30)

    def test_original_module_untouched(self):
        import inspect

        source = inspect.getsource(cellcontext.run)
        self.assertIn("min((cc for cc, pp in context_effect if pp == p), key=lambda cc: abs(cc-c))", source)

    def test_report_is_reproducible(self):
        a = summarize(TUNING_SEEDS[:5])
        b = summarize(TUNING_SEEDS[:5])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
