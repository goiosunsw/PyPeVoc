import unittest
import numpy as np

import pypevoc.speech as sp

def generate_recursive_noise(a = [-1], length=1000):
    x = np.random.rand(length)

    #y = np.zeros(len(x)-len(a_req)-1)
    y = [0] * len(a)
    for xx in x:
        yy = xx
        for ii, aa in enumerate(a):
            yy += -aa*y[-ii-1]
        y.append(yy)

    return np.array(y)


class test_lpc(unittest.TestCase):
    def test_coef_length(self):
        a_req = [-.5,.25]
        y = generate_recursive_noise(a=a_req)
        order = len(a_req)
        a_pyp = sp.analysis.lpc(y,order=order)

        self.assertEqual(order, len(a_pyp))


    def test_coef_equivalence(self):
        a_req = [-.5,.25]
        y = generate_recursive_noise(a=a_req)
        order = len(a_req)
        a_pyp = sp.analysis.lpc(y,order=order)

        for ap, at in zip(a_pyp, a_req):
            self.assertAlmostEqual(ap, at, delta=0.0001)

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
            self.assertAlmostEqual(ap, at, delta=0.01)

