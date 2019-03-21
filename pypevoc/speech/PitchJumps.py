#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PitchJumps.py
#
#  Detect pitch jumps in vocal glides
#  
#  Copyright 2017 Andre Almeida <a.almeida@unsw.edu.au>
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

import os
import sys
import numpy as np
#import matplotlib.pyplot as pl
import pandas
from .. import PV
from .SpeechChunker import SilenceDetector
from scipy.stats import ttest_ind
from scipy.signal import argrelmax


try:
    from scipy.io.wavfile import read as wavread
    from scipy.io.wavfile import write as wavwrite
except ImportError:
    sys.stderr.write('Scipy wav reader not found!\nUsing internal reader\n')
    from AudioInterface import wavLoad as wavread
    from AudioInterface import wavWrite as wavwrite

def nextpow2(x):
    return int(2**np.ceil(np.log2(x)))


def zscore_wind(x, wleft=5, wright=5, hop=None, kind='mean'):
    if hop is None:
        hop = 1

    zs = np.zeros(len(x))
    for ii in range(wleft,len(x)-wright,hop):
        xx = x[ii-wleft:ii+wright]
        if kind=='mean':
            mx = np.nanmean(xx)
            sx = np.std(xx)
        elif kind == 'median':
            mx = np.nanmedian(xx)
            sx = np.percentile(xx,75)-np.percentile(xx,25)
        zs[ii] = (x[ii]-mx)/sx
    return zs


def linreg_err(t, x, wleft=5, wright=5, hop=None):
    if hop is None:
        hop = 1

    zs = np.zeros(len(x))
    if wright < 0:
        wm = 0
    else:
        wm=wright
    for ii in range(wleft,len(x)-wm,hop):
        xx = x[ii-wleft:ii+wright]
        tt = t[ii-wleft:ii+wright]
        
        p = np.polyfit(tt,xx,1)
        resid = xx-np.polyval(p,tt)
        std = np.std(resid)
        zs[ii] = (x[ii]-np.polyval(p,t[ii]))/std
    return zs

    
def linreg2_err(t, x, wleft=5, wright=5, hop=None, use_l=True, use_r=True):
    if hop is None:
        hop=1
        
    zs = np.zeros(len(x))
    if wright<0:
        wm = 0
    else:
        wm=wright
    for ii in range(wleft,len(x)-wm,hop):
        ts=[]
        xl = x[ii-wleft:ii]
        tl = t[ii-wleft:ii]
        xr = x[ii:ii+wright]
        tr = t[ii:ii+wright]

        if use_l>0:
            pl = np.polyfit(tl,xl,1)
            residll = xl-np.polyval(pl,tl)
            stdll = np.std(residll)
            residlr = xr-np.polyval(pl,tr)
            stdlr = np.std(residlr)
            ttl,pvl = ttest_ind(residll,residlr)
            ts.append(ttl)

        if use_r>0:
            pr = np.polyfit(tr,xr,1)
            residrr = xr-np.polyval(pr,tr)
            stdrr = np.std(residrr)
            residrl = xl-np.polyval(pr,tl)
            stdrl = np.std(residrl)
            ttr,pvr = ttest_ind(residrr,residrl)
            ts.append(-ttr)
        zs[ii] = np.mean(ts)
    return zs


def avg_interpolator(tn, t, x, twind=0):
    xn = np.zeros(len(tn))
    for ii, tt in enumerate(tn):
        try:
            ior = np.flatnonzero(t > tt+twind)[0]
        except IndexError:
            ior = len(t)
        try:
            iol = np.flatnonzero(t < tt-twind)[-1]
        except IndexError:
            iol = 0

        xn[ii] = np.mean(x[iol:ior])
    return xn


class JumpDetector(object):
    def __init__(self, min_freq=70,
                 pitch_t_hop=0.02,
                 regressor_t=0.5,
                 t_threshold=10,
                 mag_threshold=0.01):
        """
        Pitch jump detector object,

        Calculates pitch track and detects jumps by comparing linear
        trends on each side of a smaple

        Arguments:
        * min_freq: minimum frequency for pitch detector
        * pitch_t_hop: time between pitch estimates
        * regressor_t: time for estimation of linear slopes
                       in pitch track
        * t_threshold: threshold for t-test comparator
        * mag_threshold: magnitude threshold for pitch track
        """
        self.min_freq = min_freq
        self.pitch_t_hop = pitch_t_hop
        self.mag_threshold = mag_threshold
        self.t_threshold = t_threshold
        self.regressor_t = regressor_t
        self.slope_t = regressor_t

    def detect_pitch(self, w, sr):
        nfft = nextpow2(sr/self.min_freq*2)
        n_hop = nextpow2(sr*self.pitch_t_hop)
        pv = PV(w, sr, nfft=nfft, hop=n_hop)
        pv.run_pv()
        self.mag = np.sqrt(np.sum(pv.mag**2, axis=1))
        self.t = pv.get_time_vector()
        self.f0 = pv.calc_f0()
        self.nfft = nfft
        self.nhop = n_hop

    def detect_jumps(self):
        wle = int(self.regressor_t/self.pitch_t_hop)
        isel = self.mag > np.max(self.mag)*self.mag_threshold
        tsel = self.t[isel]
        fsel = self.f0[isel]
        self.isel = isel
        #pl.plot(np.flatnonzero(isel),20*np.log10(m[isel]))

        le = linreg2_err(tsel, fsel, wleft=wle, wright=wle, use_l=True)
        #ax[0].plot(tsel,fsel)
        #ax[1].plot(tsel,le)
        
        imax = argrelmax(le)[0]
        lemax = le[imax]
        idx = imax[lemax > self.t_threshold]
        ijup = idx
        #ax[0].plot(tsel[idx],fsel[idx],'o')
        #ax[1].plot(tsel[idx],le[idx],'o')
        
        imin = argrelmax(-le)[0]
        lemin = le[imin]
        idx = imin[lemin < -self.t_threshold]
        ijdn = idx
        #ax[0].plot(tsel[idx],fsel[idx],'o')
        #ax[1].plot(tsel[idx],le[idx],'o')

        self.down_jump_indices = np.asarray(ijdn)
        self.up_jump_indices = np.asarray(ijup)
        self.down_jump_times = tsel[ijdn]
        self.up_jump_times = tsel[ijup]

    def calc_jump_params(self):
        tsel = self.t[self.isel]
        fsel = self.f0[self.isel]
        ijup = self.up_jump_indices
        ijdn = self.down_jump_indices
 
        nsl = int(self.slope_t/self.pitch_t_hop)

        alli = np.sort(np.concatenate((ijup, ijdn)))
        # pl.figure()
        #pl.plot(tsel,fsel)
        p = []
        intcpts = []
        sumres = []
        for ii in alli:
            il = max(0, ii-nsl)
            ir = min(ii+nsl, len(fsel))
            polyl = np.polyfit(tsel[il:ii], fsel[il:ii], 1)
            intl = np.polyval(polyl, tsel[ii])
            rsuml = np.sqrt(np.nansum((fsel[il:ii]-np.polyval(polyl, tsel[il:ii]))**2)/(ii-il))

            polyr = np.polyfit(tsel[ii+1:ir], fsel[ii+1:ir], 1)
            intr = np.polyval(polyr, tsel[ii])
            rsumr = np.sqrt(np.nansum((fsel[ii+1:ir]-np.polyval(polyr, tsel[ii+1:ir]))**2)/(ir-ii))

            #pl.plot(tsel[il:ii+1],np.polyval(polyl,tsel[il:ii+1]),color='r',alpha=.5)
            #pl.plot(tsel[ii:ir],np.polyval(polyr,tsel[ii:ir]),color='m',alpha=.5)

            p.append([polyl, polyr])
            intcpts.append([intl, intr])
            sumres.append([rsuml, rsumr])

        self.intcpts = np.array(intcpts)
        self.sumres = np.array(sumres)

        #pl.plot(tsel[alli],intcpts[:,0],'o')
        #pl.plot(tsel[alli],intcpts[:,1],'o')

    def process(self, w, sr):
        """
        process pitch tracking and jump detection
        """
        self.detect_pitch(w, sr)
        self.detect_jumps()
        self.calc_jump_params()
        return np.sort(np.concatenate([self.up_jump_times,
                                       self.down_jump_times]))

    def get_jump_table(self):
        allt = np.sort(np.concatenate([self.up_jump_times,
                                       self.down_jump_times]))

        df = pandas.DataFrame({'segment_time': allt,
                               'f_before': self.intcpts[:, 0],
                               'f_after': self.intcpts[:, 1],
                               'f_cent': np.mean(self.intcpts, axis=1),
                               'df': (np.diff(self.intcpts, axis=1))[:, 0],
                               'residue_before': self.sumres[:, 0],
                               'residue_after': self.sumres[:, 1],
                               'residue_total': np.sum(self.sumres, axis=1)})

        return df


def segment_and_detect_jumps(w, sr, **kwargs):
    sc = SilenceDetector(w, sr, fmin=50, fmax=1000)
    jd = JumpDetector(**kwargs)
    df = pandas.DataFrame()
    for ii, (tst, tend) in enumerate(zip(sc.tst, sc.tend)):
        ww = w[int(tst*sr):int(tend*sr)]
        tjmp = jd.process(ww, sr)
        try:
            dfi = jd.get_jump_table()
            dfi['rec_time'] = dfi['segment_time']+tst
            dfi['region_nbr'] = ii
            df = df.append(dfi, ignore_index=True)
        except IndexError:
            sys.stderr.write("Jump table empty between {:.2f} and {:.2f}\n".format(tst,tend))

    segments = pandas.DataFrame({'nbr': np.arange(len(sc.tst)), 'start': sc.tst,
                                 'end': sc.tend})
    return df, segments

def file_reader(filename, chan):
    file_base, file_ext = os.path.splitext(filename)
    if file_ext.lower() == '.aup':
        import audacity
        aud = audacity.Aup(filename)
        w = aud.get_channel_data(chan)
        sr = aud.rate
    else:
        sys.stderr.write("Format not recognized: %s" % file_ext)
        return
    return(sr, w)


def pitch_jump_file(filename, channel_nbr=0):
    sr, w = file_reader(filename, channel_nbr)
    df,dfs = segment_and_detect_jumps(w, sr)
    df.to_csv('pitch_jumps.csv')
    dfs.to_csv('segments.csv')
