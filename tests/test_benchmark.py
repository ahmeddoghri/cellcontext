import unittest
import cellcontext

class BenchmarkTest(unittest.TestCase):
    def test_research_method_clears_baseline(self):
        result = cellcontext.run()
        self.assertGreaterEqual(result["mae_reduction_pct"], 30)

if __name__ == "__main__":
    unittest.main()
