import unittest
import numpy as np

from Periodicity import *

def gen_sin(f = 440, sr=48000, nsamp=4800):
    #nsamp = int(dur*sr)
    return np.sin(2.*np.pi*float(f)/sr*np.arange(nsamp))

class testPeriodicity(unittest.TestCase):
    def test_single_period_sin_xcorr(self):
        f0 = 500.
        sr = 48000
        nsam = 4800
        x = gen_sin(f=f0, sr=sr, nsamp=nsam)
        pts = PeriodTimeSeries(x,sr=sr, method='xcorr')
        pts.per_at_index(nsam/2)
        period = pts.periods[0]
        p0=period.get_preferred_period()
        self.assertAlmostEqual(sr/p0,f0,delta=1.0)
    
    def test_preferred_period_is_scalar(self):
        x = gen_sin()
        nsam = len(x)
        pts = PeriodTimeSeries(x,method='xcorr')
        pts.per_at_index(nsam/2)
        period = pts.periods[0]
        p0=period.get_preferred_period()
        self.assertIsInstance(p0,float)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
