# pypevoc.speech.glottal.py
#
# Part of PyPeVoc python package
#
# Copyright (C) 2018 Andre Almeida
#
# based on covarep's IAIF:
# https://github.com/covarep/covarep/blob/master/glottalsource/iaif.m
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import scipy.signal as sig
from .SpeechAnalysis import lpc


class PaddedFilter(object):
    def __init__(self, input_signal,
                 n_before=0, n_after=0,
                 mode='zeros'):
        # Padded filter object, applies filters to a signal
        # while first padding on left and/or right
        self.mode = mode
        self.n_before = n_before
        self.n_after = n_after
        self.input_signal = input_signal

    @property
    def input_signal(self):
        return self._input_signal

    @input_signal.setter
    def input_signal(self, x):
        self._input_signal = x
        if self.mode == 'ramp':
            pad_before = np.linspace(-x[0],x[0], self.n_before)
            pad_after = np.linspace(x[-1],-x[-1], self.n_after)
        else:
            pad_before = np.zeros(self.n_before)
            pad_after = np.zeros(self.n_after)
        self._padded_input = np.concatenate((pad_before, x, pad_after))
        self._padded_output = self._padded_input

    @property
    def output_signal(self):
        if self.n_after:
            return self._padded_output[self.n_before:-self.n_after]
        else:
            return self._padded_output[self.n_before:]

    def apply_fir(self, b):
        self._padded_output = sig.lfilter(b, [1], self._padded_input)
        return self.output_signal

    def apply_fir_to_last_output(self, b):
        self._padded_output = sig.lfilter(b, [1], self._padded_output)
        return self.output_signal


def fir_pre_phase(b, x, n_ramp=None):
    # applies a FIR filter with a pre-phase ramp
    # to reduce ripple
    #
    # Arguments:
    # * b: FIR coefficients
    # * x: input signal
    # * n_ramp: number of samples in pre-ramp 
    #           (default = len(b))
    signal = np.concat((np.linspace(-x[0],x[0], n_ramp), x))
    y = np.lfilter(b,1,signal)
    return y[n_ramp+1:]


class InverseFilter:
    # implements an inverse filter object
    # based on Alku's IAIF
    #
    # P. Alku, "Glottal wave analysis with pitch synchronous iterative
    # adaptive inverse filtering", Speech Communication, vol. 11, no. 2-3,
    # pp. 109–118, 1992.
    def __init__(self, Fs=1, tract_order=None, glottal_order=None,
            leaky_integration=0.99, hpfilt=1):
        # Initialise an inverse filter object
        #
        # Fs:                sample rate (default 1)
        # tract_order:       order for Vocal Tract LPC
        #                    (default: Fs/1000 + 4)
        # glottal_order:     order fot Glottal Source LPC
        #                    (default: Fs/2000)
        # leaky_integration: leaky integration coef
        # hpfilt:            number of high pass filters to apply

        if tract_order is None:
            tract_order = 2*int(np.round(Fs/2000))

        if glottal_order is None:
            glottal_order = 2*int(np.round(Fs/4000))+4

        if hpfilt_in is None:
            hpfilt_in = np.array([])

        self.Fs = Fs
        self.tract_order = tract_order
        self.glottal_order = glottal_order
        self.leaky_integration = leaky_integration
        self.hpfilt = hpfilt
        self.pre_filter = tract_order+1
        self.init_preliminary_filter()

    def init_preliminary_filter(self, order=None, freq_stop=40, freq_pass=70):
        # calculate filter coefficients for preliminary hp filter
        if order is None:
            order = int(np.round(300/16000*self.Fs))
        self.hpfilt_b = sig.firls(order,
                                  [0, freq_stop, freq_pass, self.Fs/2],
                                  [0, 0, 1, 1],
                                  [1, 1])

    def preliminary_filter(self, x):
        # High pass filter signal to remove possible LF fluctuations
        y = x.copy()
        n_pad = int(np.round(len(self.hpfilt_b)/2-1))
        for ii in range(self.hpfilt):
            y = np.concatenate((y,np.zeros(n_pad)))
            y = sig.lfilter(self.hpfilt_b, 1, y)
            y = y[n_pad:]
        return y

    def source_filter_estimation(self, x):
        # Calculates the source and filter parameters
        # (independent of preliminary hp filter)
        # - Combined effect of lip radiation and glottal flow

        assert len(x) > self.tract_order

        y = self.pre_phase_ramp(x)
        y = self.glottal_first_estimate(y)
        self.vocal_first_estimate()
        y = self.glottal_second_estimate(y)
        self.vocal_estimate()

    def glottal_estimate(self,x):
        # estimates the effect of the vocal tract
        # and cancels it through inverse filtering
        # 
        # returns 
        win = np.hanning(len(x))
        Hg = sa.lpc(x*win,1)
        #self.hg1 = Hg
        # filter with pre_phase ramp
        y = fir_pre_phase(Hg1,1,signal)

        return hg, y 
    
    def vocal_estimate(self,x):
        Hvt = sa.lpc(y*win, self.vocal_order)
        g = fir_pre_phase(Hvt, 1, x)
        


