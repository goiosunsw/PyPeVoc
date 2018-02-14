import unittest
import numpy as np

from pypevoc.Periodicity import period_marks_corr, PeriodTimeSeries


def gen_sin(f=440, sr=48000, nsamp=4800):
    # nsamp = int(dur*sr)
    return np.sin(2.*np.pi*float(f)/sr*np.arange(nsamp))


class testPeriodicity(unittest.TestCase):
    def test_single_period_sin_xcorr(self):
        f0 = 500.
        sr = 48000
        nsam = 4800
        x = gen_sin(f=f0, sr=sr, nsamp=nsam)
        pts = PeriodTimeSeries(x, sr=sr, method='xcorr')
        period = pts.per_at_index(nsam/2)
        p0 = period.get_preferred_period()
        self.assertAlmostEqual(sr/p0, f0, delta=1.0)

    def test_preferred_period_is_scalar(self):
        x = gen_sin()
        nsam = len(x)
        pts = PeriodTimeSeries(x, method='xcorr')
        period = pts.per_at_index(nsam/2)
        p0 = period.get_preferred_period()
        self.assertIsInstance(p0, float)


class testPeriodMarks(unittest.TestCase):
    def test_period_mark_corr_int_samples_per_period(self):
        sr = 1.0
        f0 = sr/8
        nsam = 1024
        x = gen_sin(f=f0, sr=sr, nsamp=nsam)
        marks = period_marks_corr(x, sr=sr, tf=[0, nsam],
                                  f=[f0, f0], window_size=256)
        period = 1./f0
        dmarks = np.diff(marks[1:])
        for dm in dmarks:
            self.assertAlmostEqual(dm, period)

    def test_period_mark_corr_frac_samples_per_period(self):
        sr = 1.0
        f0 = sr/64.3
        nsam = 1024
        x = gen_sin(f=f0, sr=sr, nsamp=nsam)
        marks = period_marks_corr(x, sr=sr, tf=[0, nsam],
                                  f=[f0, f0], window_size=256)
        period = 1./f0
        dmarks = np.diff(marks[1:])
        for dm in dmarks:
            self.assertAlmostEqual(dm, period, places=1)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
