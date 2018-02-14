import unittest
import numpy as np

import pypevoc.speech as sp

def generate_recursive_noise(a = [-1], length=1000):
    x = np.random.rand(length)
    for aa in a:
        x += aa*x
    return x


class test_lpc(unittest.TestCase):
    def compare_to_talkbox(self):
        try:
            import scikits.talkbox as tbox
        except ImportError:
            return
        
        order = 1
        y = generate_recursive_noise()
        a_pyp = sp.analysis.lpc(y,order=order)
        a_tbx = tbox.lpc(y,order=order)

        for ap, at in zip(a_pyp, a_tbx):
            assertAlmostEqual(ap, at, delta=0.01)

