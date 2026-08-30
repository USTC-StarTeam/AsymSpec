import math
import unittest


def kl(p, q):
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def jsd(p, q):
    midpoint = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


class CDABoundTest(unittest.TestCase):
    def test_identical_distributions_keep_base_threshold(self):
        gamma = 0.5
        divergence = jsd([0.2, 0.8], [0.2, 0.8])
        self.assertAlmostEqual(gamma * math.exp(-divergence), gamma)

    def test_binary_jsd_implies_paper_bound(self):
        gamma = 0.5
        probes = [
            ([1.0, 0.0], [0.0, 1.0]),
            ([0.9, 0.1], [0.2, 0.8]),
            ([0.5, 0.5], [0.5, 0.5]),
        ]
        for p, q in probes:
            divergence = jsd(p, q)
            gamma_eff = gamma * math.exp(-divergence)
            self.assertGreaterEqual(divergence, 0.0)
            self.assertLessEqual(divergence, math.log(2) + 1e-12)
            self.assertGreaterEqual(gamma_eff, gamma / 2 - 1e-12)
            self.assertLessEqual(gamma_eff, gamma + 1e-12)


if __name__ == "__main__":
    unittest.main()
