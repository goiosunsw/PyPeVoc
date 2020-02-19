#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Heterodyne.py
#
#  Synchronous decomposition of periodic signals 
#
#  Copyright 2018 Andre Almeida <andregoios@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import sys
import collections
import numpy as np

from .PVAnalysis import SinSum



def heterodyne(x, hetsig, wind=None, hop=None):
    """
    Heterodyner: calculates the complex amplitude of a sine wave centered at f

    Arguments:
        x: signal
        f: normalised frequency vector of same length as x
           (frequency/sr)
        wind: windowing function (array, defaults to 256 point rectangular)
        nhop: samples between windows (defaults to 1/2 the window length)
    """

    ret = []
    icent = []
    if wind is None:
        wind = np.ones(2**8)
    wlen = len(wind)
    wnorm = np.sum(wind)
    #fvec[np.logical_not(np.isfinite(fvec))]=0
    xf = x*hetsig
    for ii in range(0,len(x)-wlen,hop):
        xx = xf[ii:ii+wlen]
        xw = xx*wind
        ret.append(np.sum(xw)/wnorm)
        icent.append(ii+wlen//2)
    return np.array(ret)*2,np.array(icent)



class Heterodyne(object):
    """
    Perform a sine sum decomposition based on a f0 track
    """

    def __init__(self, x, sr=1.0, nwind=1024, wfun=np.hanning, nhop=None):
        """
        Create a heterodyner object, storing the signal basic information
        of the analysis

        Arguments:
            * x:        signal
            * sr:       sampling rate
            * nwind:    default window length 
            * wfun:     default windowing function
            * ampthr:   amplitude threshold for filtering in resynthesis
        """
        self.x = x
        self.sr = sr
        self.nwind = nwind
        self.nhop = nhop
        self.nsamp = len(x)
        self.wfun = wfun
        self.ampthr = ampthr

        self._fix_params()

    def add_partial(self, f, tf=None, fidx=None,
                    wind=None, nhop=None, 
                    t=None, idx=None):
        """
        set the starting indices for the windowed analysis
        """
        self.idx = idx
        

    def harmonic_times(self, n=1):
        if self.variable_resolution:
            return self.th[n-1]
        else:
            return self.th

    def harmonic_amplitudes(self, n=1):
        if self.variable_resolution:
            return self.ah[n-1]
        else:
            return self.ah[:,n-1]

    def harmonic_frequencies(self, n=1):
        if self.variable_resolution:
            return self.f[self.idxh[n-1]]*n
        else:
            return self.f[self.idxh]*n


    def heterodyner_signal(self, n=1):
        """
        return a reference variable-frequency signal
        with frequency equal to n* the harmonic of the frequency vector
        """
        omega = self.fvec*2*np.pi*n
        phvec = np.cumsum(omega)
        return np.exp(1j*phvec)

    def set_fvec(self, f0c, th=None, adjust=False):
        tvec = np.arange(len(self.x))/self.sr

   
        if th is not None:
            fvec = np.interp(tvec, th, f0c)
        else:
            fvec = f0c

        # fix for single-frequency values
        if not isinstance(fvec, collections.abc.Sequence):
            fvec = fvec*np.ones(self.nsamp)
   
        self.fvec = fvec/self.sr
        self.fmin = max(self.fmin,min(fvec))

        if adjust:
            f0c, th = self.calc_adjusted_freq(fvec)
            self.fvec = np.interp(tvec, th, f0c)
        if self.variable_resolution:
            self.th = [[] for ii in range(self.nharm)]
            self.ah = [[] for ii in range(self.nharm)]
            self.idxh = [[] for ii in range(self.nharm)]
        else:
            self.th = np.arange(self.nwind//2, self.nsamp-self.nwind//2, self.nhop)/self.sr
            self.idxh = np.arange(self.nwind//2, self.nsamp-self.nwind//2,
                                  self.nhop).astype('i')
            self.ah = np.zeros((self.th.shape[0],self.nharm),dtype='complex')


    def extract_partial(self, n):
        """
        calculates complex amplitudes of partials
        """
        x = self.x
        hetsig = self.heterodyner_signal(n=n)
        if self.variable_resolution:
            wind = self.wfun(int(self.nper/self.fmin*self.sr/n))
        else:
            wind = self.wind
        h,th = heterodyne(x, hetsig, wind=wind, hop=self.nhop)
        return h, th

    def filter_harmonic(self, n):
        """
        mute intervals not to be taken into account in resynthesis
        """
        tvec = np.arange(self.nsamp)/self.sr
        hf = np.interp(tvec, self.harmonic_times(n), self.harmonic_amplitudes(n))
        idx = (self.f<self.fmin) | (self.f>self.fmax) | (self.f*n>self.sr/2.2)
        rmsmin = np.max(np.abs(hf))*self.ampthr
        idx = idx | (np.abs(hf)<rmsmin)
        hf[idx] = 0
        return hf

    def resynth_partial(self, n):
        """
        resynthesises a partial based on extracted complex amplitudes
        """
        th = self.harmonic_times(n)
        h = self.harmonic_amplitudes(n)
        hsig = self.heterodyner_signal(n)
        tvec = np.arange(self.nsamp)/self.sr

        hf = np.interp(tvec, th, h)
        ff = self.fvec
        hf = self.filter_harmonic(n)

        return np.real(np.conjugate(hsig)*hf)
        
    def extract_partials(self):
        """
        extracts all partials in from 1 to nharm 
        and puts them into a matrix
        """

        for ii in range(1, self.nharm+1):
            hh, th = self.extract_partial(ii)
            self.ah[:,ii-1] = hh

        return self.ah, self.th


    def resynth(self):
        x = np.zeros(self.nsamp)
        for ii in range(self.nharm):
            x+=self.resynth_partial(ii)
        return(x)

    def clone(self):
        from copy import copy
        return copy(self)




class HeterodyneHarmonic:
    """
    Perform a sine sum decomposition based on a f0 track
    """

    def __init__(self, x, sr=1.0, tf=None, f=None, nper=None, nwind=1024, nhop=None,
                 wfun=np.hanning, ampthr=0.1, nharm=5, fmin=0.1, fmax=1000, include_dc=False):
        """
        Perform a sine sum analysis on signal x with sampling rate sr.

        Arguments:
            * x:        signal
            * sr:       sampling rate
            * f:        frequency track (float or array)
            * tf:       time values of f 
                        (defaults to evenly distributed along x)
            * nwind:    window length (defaults to 3* maximum period)
            * nhop:     interval between estimations 
                        (defaults to 1/2 nwind)
            * nper:     number of periods to use in heterodyning
                        (takes precedence over nwind)
            * wfun:     windowing function
            * nharm:    number of harmonics to use in decomposition
            * ampthr:   amplitude threshold for filtering in resynthesis
            * include_dc: calculate DC amplitude
        """
        self.x = x
        self.sr = sr
        self.nper = nper
        self.nhop = nhop
        self.nwind = nwind
        self.tf = tf
        self.fvals = f
        self.nharm = nharm
        self.nsamp = len(x)
        self.wfun = wfun
        self.ampthr = ampthr
        self.fmin = fmin
        self.fmax = fmax
        self.include_dc = include_dc

        self._fix_params()
        self.extract_partials()

    @property
    def f0(self):
        return self.fvec*self.sr

        
    @property
    def f(self):
        tvec = np.arange(self.nsamp)/self.sr
        f0 = np.interp(self.t, tvec, self.f0)
        if self.include_dc:
            return np.array([f0*(n) for n in range(self.nharm)]).T
        else:
            return np.array([f0*(n) for n in range(1,self.nharm)]).T

    @property
    def angle_ratios(self):
        ang = np.angle(self.camp/np.tile(self.camp[:,:1],(1,self.camp.shape[1])))
        if self.include_dc:
            ang=np.hstack((np.zeros((ang.shape[0],1)),ang))
        return ang
    
    @property
    def partial_frequencies(self):
        dt = self.nhop/self.sr
        newf=self.f[1:,:]-np.diff(np.unwrap(np.angle(self.camp)),axis=0)/dt/2/np.pi
        if self.include_dc:
            newf=np.hstack((np.zeros((newf.shape[0],1)),newf))
        return newf

    @property
    def t(self):
        return self.th

    @property
    def camp(self):
        if self.include_dc:
            return self.ah
        else:
            return self.ah[:,1:]
    
    @camp.setter
    def camp(self, mx):
        self.ah = mx

    def _fix_params(self):
        if self.nhop is None:
            self.nhop = self.nwind//2
        self.wind = self.wfun(self.nwind)
        #print(max(tvec))
        tvec = np.arange(self.nsamp)/self.sr

        if self.nper is not None:
            self.variable_resolution = True
            self.nwind = int(round(self.nper/self.fmin))
            print('window set to %d'%self.nwind)
        else:
            self.variable_resolution = False

        self.set_fvec(self.fvals,self.tf)

    def harmonic_times(self, n=1):
        if self.variable_resolution:
            return self.th[n]
        else:
            return self.th

    def harmonic_amplitudes(self, n=1):
        if self.variable_resolution:
            return self.ah[n]
        else:
            return self.ah[:,n]

    def harmonic_frequencies(self, n=1):
        if self.variable_resolution:
            return self.f[self.idxh[n]]*n
        else:
            return self.f[self.idxh]*n


    def heterodyner_signal(self, n=1):
        """
        return a reference variable-frequency signal
        with frequency equal to n* the harmonic of the frequency vector
        """
        return self.heterodyner_signal_from_f(self.fvec*n)

    def heterodyner_signal_from_f(self,f):
        """
        calculate a heterodyner signal based on the normalised
        frequency vector

        input a normalised frequency vector
        """
        omega = f*2*np.pi
        phvec = np.cumsum(omega)
        return np.exp(1j*phvec)


    def calc_adjusted_freq(self, fvec, nwind=None, nhop=None):
        """
        adjusts the frequency track accordin to a first-pass heterodyne 
        analysis. 

        ARguments:
            * nwind:        object nwind
            * nhop:         object nhop
        """

        x = self.x
        sr = self.sr
        tvec = np.arange(len(x))/sr
        hetsig = self.heterodyner_signal_from_f(fvec)
        if nwind is None:
            wind = self.wind
        else:
            wind = self.wfun(nwind)

        if nhop is None:
            nhop = len(wind)//2

        h, ih = heterodyne(x, hetsig, wind=wind, hop=nhop)
        th = ih/sr
        dph = np.concatenate(([0], np.diff(np.unwrap(np.angle(h)))))
        f0c = np.interp(th,tvec,fvec)-dph/(nhop)/2/np.pi 
        return f0c, th

    def set_fvec(self, f0c, th=None, adjust=False):
        tvec = np.arange(len(self.x))/self.sr

        if th is not None:
            fvec = np.interp(tvec, th, f0c)
        else:
            fvec = f0c

        # fix for single-frequency values
        if not isinstance(fvec, collections.abc.Sequence):
            fvec = fvec*np.ones(self.nsamp)

        self.fvec = fvec/self.sr
        self.fmin = max(self.fmin,min(fvec))

        if adjust:
            f0c, th = self.calc_adjusted_freq(fvec)
            self.fvec = np.interp(tvec, th, f0c)
        if self.variable_resolution:
            self.th = [[] for ii in range(self.nharm)]
            self.ah = [[] for ii in range(self.nharm)]
            self.idxh = [[] for ii in range(self.nharm)]
        else:
            self.th = np.arange(self.nwind//2, self.nsamp-(self.nwind-self.nwind//2), self.nhop)/self.sr
            self.idxh = np.arange(self.nwind//2, self.nsamp-self.nwind//2,
                                  self.nhop).astype('i')
            self.ah = np.zeros((self.th.shape[0],self.nharm),dtype='complex')


    def extract_partial(self, n):
        """
        calculates complex amplitudes of partials
        """
        x = self.x
        hetsig = self.heterodyner_signal(n=n)
        if self.variable_resolution:
            wind = self.wfun(int(self.nper/self.fmin*self.sr/n))
        else:
            wind = self.wind
        h,th = heterodyne(x, hetsig, wind=wind, hop=self.nhop)
        return h, th

    def filter_harmonic(self, n):
        """
        mute intervals not to be taken into account in resynthesis
        """
        tvec = np.arange(self.nsamp)/self.sr
        hf = np.interp(tvec, self.harmonic_times(n), self.harmonic_amplitudes(n))
        idx = (self.f0<self.fmin) | (self.f0>self.fmax) | (self.f0*n>self.sr/2.2)
        rmsmin = np.max(np.abs(hf))*self.ampthr
        idx = idx | (np.abs(hf)<rmsmin)
        hf[idx] = 0
        return hf

    def resynth_partial(self, n):
        """
        resynthesises a partial based on extracted complex amplitudes
        """
        th = self.harmonic_times(n)
        h = self.harmonic_amplitudes(n)
        hsig = self.heterodyner_signal(n)
        tvec = np.arange(self.nsamp)/self.sr

        hf = np.interp(tvec, th, h)
        ff = self.fvec
        hf = self.filter_harmonic(n)

        return np.real(np.conjugate(hsig)*hf)
        
    def get_voice_component(self, x, sr, f, nharm, nfft=2**13, hop=2**11, ampthr=0.1, fmin=70, fmax=500):
        xs = np.zeros(len(x))
        tvec = np.arange(len(x))/sr
        
        idx = np.isnan(f)
        fc = f
        fc = f[np.logical_not(idx)]
        tf = tvec[np.logical_not(idx)]
        
        for nn in range(nharm):
            hno = nn+1
            fh = fc*hno
            h,th,hsig = heterodyne(x,sr,tf=tf,f=fh,wlen=nfft,hop=hop)
            hf = np.interp(tvec,th,h)
            #ff = f0f*hno
            hf[idx]=0
            xs += np.real(np.conjugate(hsig)*hf)
            
        return xs

    def extract_partials(self):
        """
        extracts all partials in from 1 to nharm 
        and puts them into a matrix
        """

        for ii in range(0, self.nharm):
            hh, th = self.extract_partial(ii)
            self.ah[:,ii] = hh
        self.ah[:,0] /= 2

        return self.ah, self.th

    def resynth(self):
        x = np.zeros(self.nsamp)
        for ii in range(self.nharm):
            x+=self.resynth_partial(ii)
        return(x)

    def clone(self):
        from copy import copy
        return copy(self)



