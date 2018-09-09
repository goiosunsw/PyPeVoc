import numpy as np
import sys
import pypevoc.PeakFinder as pf

import unittest

def parabolic_peak(max_pos=1.0, max_val=1.0, n=3, a=-1.):
    x = np.arange(n)
    b = -max_pos * 2 * a
    c = max_val - a * max_pos * (b + max_pos)
    return a*x*x + b*x + c


class testPeakFinder(unittest.TestCase):
    def testFindOnePeak(self):
        x = np.linspace(0, 1, 10)
        x = np.concatenate((x, np.linspace(.9, 1, 9)))
        peaks = pf.PeakFinder(x)
        assert(len(peaks.pos) == 1)
        self.assertEqual(peaks.pos, 9)

    def test_refine_one_peak_centered(self):
        x = parabolic_peak(max_pos=1.0)
        peaks = pf.PeakFinder(x)
        peaks.refine_all()
        assert(len(peaks.pos) == 1)
        self.assertEqual(peaks.pos, 1.0)

    def test_refine_one_peak_at_random_pos(self):
        mypos = 1.2
        x = parabolic_peak(max_pos=mypos, n=4)
        peaks = pf.PeakFinder(x)
        peaks.refine_all()
        self.assertListEqual(peaks.fpos.tolist(), [mypos])

    def test_refine_one_peak_between_samples(self):
        x = parabolic_peak(max_pos=1.5, n=4)
        peaks = pf.PeakFinder(x)
        peaks.refine_all()
        self.assertListEqual(peaks.fpos.tolist(), [1.5])

    def test_refine_one_peak_almost_between_samples(self):
        mypos = 1.499
        x = parabolic_peak(max_pos=mypos, n=4)
        peaks = pf.PeakFinder(x)
        peaks.refine_all()
        self.assertEqual(len(peaks.fpos), 1)
        self.assertAlmostEqual(peaks.fpos[0], mypos)



def main():
    unittest.main()

if __name__ == '__main__':
    main()
