
# pypevoc.speech.DAP.py
#
# Part of PyPeVoc python package
#
# Copyright (C) 2018 Andre Almeida
#
# based on covarep's env_dap.m:
# https://github.com/covarep/covarep/blob/master/envelope/env_dap.m
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
import scipy.linalg as sla
import logging

class EnvelopeDAP(object):
    def __init__(self, sr=1.0, order=4, dftlen=2**12, maxit=50, alpha=.5, dISthresh=1e-6,
                 minbw=None):
        self.sr = sr
        self.order = order
        self.dftlen = dftlen
        self.maxit = maxit
        self.alpha = alpha
        self.dISthresh = dISthresh
        if minbw is not None:
            self.minrr = np.exp(-np.pi/self.sr*minbw) 

    def estimate(self, freqs, amps, order=None):
        if order is None:
            order = self.order
        omegas = 2*np.pi*freqs/self.sr
        amps = np.abs(amps)
        nharm = len(amps)

        # imaginary part of z variable
        ejw = np.exp(-1j*omegas * np.arange(0,order+1))
        inv_ejw = np.exp(1j*omegas * np.arange(0,order+1))

        # target autocorr matrix
        r = 1/nharm*np.real(amps**2*inv_ejw)
        rmx_inv = sla.inv(sla.toeplitz(r))

        # initial guess (LPC)
        use_r = r[:order]
        a = sla.solve_toeplitz(r[:-1],-r[1:])
        # calculate the prediction error
        
        

    

