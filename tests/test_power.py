import unittest
from scipy.stats import norm
from power_analysis import power, required_pairs


class PowerTests(unittest.TestCase):
    def test_null_rejection_probability(self):
        self.assertAlmostEqual(power(20, .32, gap=0), .05, places=10)

    def test_smallest_sufficient_sample(self):
        for sd in (.20, .32, .53):
            n = required_pairs(sd)
            self.assertLess(power(n - 1, sd), .8)
            self.assertGreaterEqual(power(n, sd), .8)

    def test_large_sample_normal_limit(self):
        n, sd, gap = 10000, .32, .005
        ncp = gap * n**.5 / sd
        normal_power = norm.sf(norm.ppf(.975) - ncp) + norm.cdf(-norm.ppf(.975) - ncp)
        self.assertAlmostEqual(power(n, sd, gap), normal_power, places=3)


if __name__ == "__main__":
    unittest.main()
