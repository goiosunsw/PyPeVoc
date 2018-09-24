import unittest
import numpy as np

import pypevoc.speech.glottal as gl

class testPaddedFilter(unittest.TestCase):
    def test_output_same_padding_as_input(self):
        filter = np.array([-1,1])
        n_pad_before = 8
        n_signal = 16
        x = np.zeros(n_signal)
        glx = gl.PaddedFilter(x,n_before=n_pad_before)
        y = glx.apply_fir(filter)
        self.assertEqual(len(x),len(y))

    def test_private_buffer_1d(self):
        glx = gl.PaddedFilter(np.zeros(10),n_before=8)
        self.assertEqual(len(glx._padded_input.shape),1)

    def test_private_output_buffer_1d(self):
        glx = gl.PaddedFilter(np.zeros(10),n_before=8)
        self.assertEqual(len(glx._padded_output.shape),1)


if __name__ == '__main__':
    unittest.main()
