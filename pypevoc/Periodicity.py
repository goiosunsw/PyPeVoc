#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Periodicity.py
#
#  Utilities for frequency and periodicity estimation
#  * Fundamental frequency estimator
#  * Tonal character
#
#
#  Copyright 2014 Andre Almeida <goios@AndreUbuntu>
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
import numpy as np
import pylab as pl
from matplotlib.colors import hsv_to_rgb
from .PeakFinder import PeakFinder as pf
from .ProgressDisplay import Progress
#from AMDF import amdf

def amdf(x, mindelay=0, maxdelay=None):
    nx = len(x)
    if maxdelay is None:
        maxdelay = nx

    y = np.zeros(nx)
    for i in range(mindelay, maxdelay):
        n = nx - i
        y[i] = (np.abs(x[0:nx-i]-x[i:])).sum()/n

    return y

# I will try to update this object so that data required for
# the initialisation of every instance stays in the caller.
# Thee caller passes itself as argument to the callee


class Periodicity(object):
    """Single period object, including multiple periodicity candidates
    """
    def __init__(self, parent, index=0):
        """Calculate the periodicity estimation for a window
           of a time signal

        Arguments:
        parent: parent object contaigning entire signal
        idx:    index of local peridoicity calulation
        """

        self.parent = parent
        self.nwind = parent.nwind
        self.wnorm = parent.wnorm
        self.wind = parent.wind
        self.sr = parent.sr

        self.mindelay = parent.mindelay
        if parent.maxdelay is None:
            self.maxdelay = int(round(self.nwind/2))
        else:
            self.maxdelay = int(parent.maxdelay)

        self.method = parent.method
        self.threshold = parent.threshold
        self.vthresh = parent.vthresh
        self.ncand = parent.ncand
        self.fftthresh = parent.fftthresh

        # Arrays with probable candidate periodicity and corresponding
        # strength
        self.cand_period = np.array([])
        self.cand_strength = np.array([])
        # Index of preferred candidate
        self.preferred = 0

        self.cand_method = parent.cand_method
        self.index=index

        self._calc()

    def _calc(self):
        """Calculate the periodicity candidates

        Arguments:
        xw: the windowed portion of time signal where periodicity
            is to be estimated
        """
        nwleft = int(np.floor(self.nwind/2))
        nwright = int(self.nwind - nwleft)
        idx = int(np.round(self.index))
        ist = idx - nwleft
        iend = idx + nwright

        xs = self.parent.x[ist:iend]
        xw = (xs-np.mean(xs)) * self.wind

        nwind = self.nwind

        # unvoiced
        pkpos = np.array([])
        pkstr = np.array([])

        peaks = None

        try:
            if self.method is 'amdf':
                xc = amdf(xw)

                maxxc = max(xc[nwind-1-self.maxdelay:nwind-1+self.maxdelay])
                xcn = (maxxc-xc)/maxxc
                imin = self.mindelay
                xcpos = xcn[imin:self.maxdelay]
                xcth = self.threshold

            elif self.method is 'xcorr':

                xc = np.correlate(xw, xw, "full") / self.wnorm

                negvals = np.flatnonzero(xc[nwind-1:] < 0)
                if len(negvals) > 0:
                    firstneg = np.min(negvals)
                else:
                    firstneg = self.mindelay
                imin = max(firstneg, self.mindelay)
                xcn = xc/max(xc[nwind-1-self.maxdelay:nwind-1+self.maxdelay])
                xcpos = xcn[nwind-1+imin:nwind-1+self.maxdelay]

                xcth = self.threshold

                # print "In xcorr. max %f, thr %f"%(max(xcpos),xcth)

            if len(xcpos) > 0 and max(xcpos) > self.vthresh:
                # this is equivlent to finding minima
                # below the absolute minimum * threshold
                peaks = pf(xcpos, minval=xcth,
                                      npeaks=self.ncand)

                peaks.refine_all()
                # peaks.plot()

                pkpos = peaks.pos + imin
                pkstr = peaks.val

                # keep = pkpos<self.maxdelay
                # pkpos = pkpos[keep]
                # pkstr = pkstr[keep]

        except IndexError as e:
            print(e)

        if len(pkpos) > 0:
            self.cand_period = pkpos
            self.cand_strength = pkstr

            if self.cand_method == 'fft':
                xf = np.fft.fft(xw)
                fftpeaks = pf(np.abs(xf[0:int(self.nwind/2)]),
                                         npeaks=self.ncand)
                # periodicity corresponding to fft peaks:
                fpos = fftpeaks.pos
                fval = fftpeaks.val
                fposkeep = fpos[fval > np.max(fval*self.fftthresh)]
                fftpkpos = self.nwind / fposkeep

                # minimum distance between correlation candidates
                # and fft peaks
                perdist = [np.min(np.abs(fftpkpos-thispos))
                           for thispos in pkpos]
                try:
                    self.preferred = np.argmin(perdist)
                except ValueError:
                    self.preferred = 0
                # print (fftpkpos)
                # print (pkpos)
            elif self.cand_method == 'min':
                self.preferred = np.argmin(pkpos)
            elif self.cand_method == 'similar':
                self.preferred = np.argmax(pkstr)
        else:
            self.preferred = 0
            # self.cand_period = np.array([np.nan])
            # self.cand_strength = np.array([np.nan])

        return xcn

    def plot_similarity(self, ax=None):

        xc = self._calc()

        if not ax:
            fig, ax = pl.subplots(1)
        ln = ax.plot(np.arange(len(xc))-self.nwind+1, xc)
        ax.hold('on')
        ax.plot(self.cand_period, self.cand_strength, 'o',
                color=ln[0].get_color())

    def set_time_properties(self, index):
        """Set the sample and time value of this periodicity estimation

        Arguments:
        index: sample index
        """

        self.index = float(index)
        self.time = float(index)/self.sr

    def sort_strength(self):
        """Sort candidates by periodicity strength

        Arguments: (None)
        """

        idx = np.argsort(self.cand_strength)[::-1]
        self.cand_period = self.cand_period[idx]
        self.cand_strength = self.cand_strength[idx]
        pref = np.flatnonzero(idx == self.preferred)
        if len(pref) > 0:
            self.preferred = pref[0]
        else:
            self.preferred = []

    def get_preferred_period(self):
        if len(self.cand_period) > 0:
            return self.cand_period[self.preferred]
        else:
            return 0

    def get_preferred_strength(self):
        if len(self.cand_period) > 0:
            return self.cand_strength[self.preferred]
        else:
            return 0


class PeriodSeries(object):
    def __init__(self, x, sr=48000, window=None, hop=None,
                 threshold = .8, vthresh = .2,
                 fmin=50, fmax=5000, 
                 ncand=8, method='xcorr',
                 cand_method='fft', fftthresh=0.1):
        """Calculate the average mean difference of x around index

        Arguments:
        x:         signal
        sr:        sample rate
        window:    window around index used for difference calculations
        threshold: ratio to lowest minima to keep as peak
        vthresh:   voicing threshold
        fmin:      value of minimum possible frequency
        fmax:      value of maximum possible frequency
        ncand:     maximum number of period candidates
        method:    type of correlation correlation / matching to use
                   'xcorr' - correlation
                   'amdf'  - average mean difference function
                   'zc'    - zero crossing
        cand_method: method for candidate selection:
                     'fft'    - based on an fft of the window
                     'min'    - minimum periodicity wins
                     'similar'- most similar wins
        fftthresh: threshold for fft peak selection (default=0.1)
        """

        self.method = method
        self.x = x.astype(float)
        self.sr = sr

        self.nx = len(x)

        if fmin is None:
            maxdelay = None
        else:
            maxdelay = int(sr/fmin)

        if fmax is None:
            mindelay = 2
        else:
            mindelay = int(sr/fmax)

        if window is None:
            if maxdelay is None:
                window = self.nx
            else:
                window = 3*maxdelay

        if not np.iterable(window):
            window = np.ones(window)

        self.wind = window
        self._calc_window_norm()

        self.nwind = len(window)
        # self.windad = amdf(window)

        self.mindelay = mindelay
        if maxdelay is None:
            self.maxdelay = int(round(self.nwind/2))
        else:
            self.maxdelay = maxdelay

        if hop is None:
            hop = self.nwind//2

        self.hop = hop

        self.method = method
        self.threshold = threshold
        self.vthresh = vthresh
        self.ncand = ncand
        self.cand_method = cand_method
        self.fftthresh = fftthresh

        # data storage
        self.periods = []

        # progress indicator
        self.progress = Progress(end=self.nx)

    def _calc_window_norm(self):
        """Calculate the normalisation function for window

        Arguments: (None)
        """

        if self.method is 'xcorr':
            w = self.wind
            self.wnorm = np.correlate(w, w, "full")
        else:
            self.wnorm = 1.

    def per_at_index(self, index):
        """Calculate the average mean difference of x around index

        Arguments:

        index:  index of x for current amdf
        threshold: ratio to lowest minima to keep as peak
        """

        pp = Periodicity(self, index)
        pp.set_time_properties(index)
        pp.sort_strength()

        # self.periods.append(pp)
        return pp

    def calc(self, hop=None, threshold=None):
        """Estimate local periodicity in the full time series

        Arguments:

        hop:       samples bewteen estimations
        threshold: peak threshold for maintaining or rejecting
                   candidates
        """

        self.periods = []
        if hop is None:
            hop = self.hop

        if threshold is not None:
            oldthresh = self.threshold
            self.threshold = threshold

        idxmax = self.nx - self.nwind
        idxvec = np.arange(self.nwind, idxmax, hop)

        sys.stderr.write("Calculating local periodicity... \n")

        for idx in idxvec:
            pp = self.per_at_index(idx)
            sys.stderr.write("\r{:6.2f}%%".format(idx*100/idxmax))
            sys.stderr.flush()
            self.periods.append(pp)

        sys.stderr.write("\ndone\n"  )

        if threshold is not None:
            self.threshold = oldthresh

    def calcPeriodByPeriod(self, threshold=None, 
                           tf=None, f=None):
        """Estimate local periodicity in the full time series

        Arguments:

        hop:       samples bewteen estimations
        threshold: peak threshold for maintaining or rejecting
                   candidates
        """

        self.periods = []
        if threshold is not None:
            oldthresh = self.threshold
            self.threshold = threshold

        # Max index for starting window
        idxmax = self.nx - self.nwind

        sys.stdout.write("Calculating local periodicity... ")
        idx = self.nwind
        while idx < idxmax:
            pp = self.per_at_index(idx)
            oldidx = idx
            if f is None:
                di = pp.get_preferred_period()
            else: 
                thisf = np.interp(pp.time, tf, f)
                if len(pp.cand_period)>0 and thisf>0:
                    imin = np.argmin(np.abs(self.sr/thisf-pp.cand_period))
                    pp.preferred = imin
                    di = pp.cand_period[imin]
                else:
                    di=0
            if di:
                idx += di
                self.periods.append(pp)
            else:
                idx += self.mindelay 

            # sys.stdout.write("\b"*15+"%6d / %6d" % (idx, self.nx))
            # sys.stdout.flush()
            self.progress.update(idx)

        self.progress.update(self.nx)
        sys.stdout.write("\ndone\n") 

        if threshold is not None:
            self.threshold = oldthresh

    def plot_candidates(self):
        """Plot a representation of candidate periodicity

        Size gives the periodicity strength, 
        color the order of preference
        """

        fig, ax = pl.subplots(2, sharex=True)

        hues = np.arange(self.ncand)/float(self.ncand)
        hsv = np.swapaxes(np.atleast_3d([[hues, np.ones(len(hues)),
                                          np.ones(len(hues))]]), 1, 2)
        cols = hsv_to_rgb(hsv).squeeze()

        for per in self.periods:
            nc = len(per.cand_period)

            ax[0].scatter(per.time*np.ones(nc), per.cand_period,
                          s=per.cand_strength*100,
                          c=cols[0:nc], alpha=.5)

        ax[0].plot(*zip(*[[per.time, float(per.get_preferred_period())]
                        for per in self.periods]), color='k')

        ax[1].plot(self.get_times(), self.get_strength())

    def get_f0(self, thresh=0.0):
        """Get f0 as a function of time

        thresh: threshod for period strength
        """

        f0 = np.zeros(len(self.periods))
        for ii, per in enumerate(self.periods):
            if per.get_preferred_strength() > thresh:
                f0[ii] = self.sr/per.get_preferred_period()
            else:
                f0[ii] = np.nan
        return f0

    def get_times(self):
        """Get f0 as a function of time
        """

        f0 = np.zeros(len(self.periods))
        for ii, per in enumerate(self.periods):
            f0[ii] = per.time
        return f0

    def get_strength(self):
        """Get f0 strength as a function of time
        """

        ss = np.zeros(len(self.periods))
        for ii, per in enumerate(self.periods):
            ss[ii] = per.get_preferred_strength()
        return ss


class PeriodTimeSeries(PeriodSeries):
    pass


class PeriodByPeriod(PeriodSeries):
    def __init__(self):
        super(PeriodByPeriod, self).__init__()

    def import_period_series(self, pts):
        """Imports a PeriodTimeSeries object

        :pts: PeriodTimeSeries object with
              time and frequency information
        :returns: None

        """
        self.f = pts.f
        self.t = pts.t
        self.sr = pts.sr

def period_marks_amdf(x, sr=1.0, t0=0.0, tf=[], f=[], window_size=1024,
                      min_per=0.001):
    """add period marks information to file,
    based on sample per sample difference between adjacent periods

    :t0: first mark position
    :window_size: window to use for comparison between periods
    :returns: TODO

    """
    marks_t = [t0]
    next_t = t0
    this_f0 = np.interp(marks_t[-1], tf, f)
    if np.isnan(this_f0):
        this_f0 = np.nanmean(f)
    period_samp = int(sr/this_f0)
    while next_t*sr < len(x) - period_samp - window_size:
        if not np.isnan(this_f0):
            period_samp = int(sr/this_f0)
            source_idx_st = int(next_t*sr)
            target_idx_st = source_idx_st + period_samp
            source_idx_end = source_idx_st + window_size
            target_idx_end = target_idx_st + window_size
            x_source = x[source_idx_st:source_idx_end]
            x_target = x[target_idx_st:target_idx_end]
            xc = amdf(x_source, x_target)
            # find max of xc near 0 lag
            # (at position window_size-1)
            peaks = pf(-xc)
            idx_min = np.argmin(np.abs(peaks.pos-window_size+1))
            delay_samp, _ = peaks.refine(idx_min)
            # delay_samp = peaks.get_pos()[idx_min]
            delay_samp -= window_size-1
            # print delay_samp
            delay_t = (-delay_samp + period_samp)/sr
            if delay_t > min_per:
                marks_t.append(next_t+(window_size+period_samp/2)/sr)
                next_t += delay_t
            else:
                next_t += 1/this_f0

        else:
            next_t = next_t + delay_t

        this_f0 = np.interp(next_t, tf, f)
    return np.array(marks_t)


def period_marks_corr(x, sr=1.0, t0=0.0, tf=[], f=[], window_size=1024,
                      min_per=0.001):
    """add period marks information to file,
    based on correlation between adjacent periods

    :t0: first mark position
    :window_size: window to use for comparison between periods
    :returns: TODO

    """
    marks_t = [t0]
    next_t = t0
    this_f0 = np.interp(marks_t[-1], tf, f)
    if np.isnan(this_f0):
        this_f0 = np.nanmean(f)
    period_samp = int(sr/this_f0)
    while next_t*sr < len(x) - period_samp - window_size:
        if not np.isnan(this_f0):
            period_samp = int(sr/this_f0)
            source_idx_st = int(next_t*sr)
            target_idx_st = source_idx_st + period_samp
            source_idx_end = source_idx_st + window_size
            target_idx_end = target_idx_st + window_size
            x_source = x[source_idx_st:source_idx_end]
            x_target = x[target_idx_st:target_idx_end]
            xc = np.correlate(x_source, x_target, "full")
            # find max of xc near 0 lag
            # (at position window_size-1)
            peaks = pf(xc)
            idx_min = np.argmin(np.abs(peaks.pos-window_size+1))
            delay_samp, _ = peaks.refine(idx_min)
            # delay_samp = peaks.get_pos()[idx_min]
            delay_samp -= window_size-1
            # print delay_samp
            delay_t = (-delay_samp + period_samp)/sr
            if delay_t > min_per:
                marks_t.append(next_t+(window_size+period_samp/2)/sr)
                next_t += delay_t
            else:
                next_t += 1/this_f0

        else:
            next_t = next_t + delay_t

        this_f0 = np.interp(next_t, tf, f)
    return np.array(marks_t)


def period_marks_peak(x, sr=1.0, tf=None, f=[], fit_points=3):
    """calculate period marks for x based on peak
    positions of the signal

    :x: signal
    :sr: sample rate (defalut 1 sample/sec)
    :tf: time at which frequency values are calulated
         (defaults to same samples as x)
    :f: frequency values
    :fit_points: number of points to use for peak fitting
    :returns: time markers
    """

    # derivative of x
    # dx = np.diff(x)

    # make sure the rate is float
    sr = float(sr)

    # build time vector for signal
    tx = np.arange(len(x))/(sr)
    # interpolate frequency values
    if tf is None:
        try:
            assert(len(f) == len(x))
        except(TypeError):
            f = f*np.ones(len(x))
    else:
        # f_orig = f
        f = np.interp(tx, tf, f)

    real_mask = np.isfinite(f)
    idx_0 = np.nonzero(real_mask)[0][0]
    period_samp = int(sr/f[idx_0])

    marks = []
    maxval = []

    # find the first minimum
    idx_start = idx_0 + np.argmin(x[idx_0:idx_0+period_samp])
    while idx_start < len(x):
        idx_end = np.min([idx_start + period_samp, len(x)])
        idx_max = np.argmax(x[idx_start:idx_end]) + idx_start

        if fit_points < 3:
            t_max = idx_max/sr
        # elif fit_points == 3:
        #    # parabolic interpolation
        else:
            # parabolic fit
            rel_idx_start = int(np.max([0,-fit_points/2]))
            rel_idx_end = np.min([rel_idx_start + fit_points,
                                 len(x) - idx_max - 1])
            # dx_fit = dx[idx_max+rel_idx_start:idx_max+rel_idx_end]
            # dx_abcissa = np.arange(rel_idx_start, rel_idx_end)+.5
            # fit_poly = np.polyfit(dx_abcissa, dx_fit, 1)
            # rel_refined_max = -fit_poly[1]/fit_poly[0]
            x_fit = x[idx_max+rel_idx_start:idx_max+rel_idx_end+1]
            x_abcissa = np.arange(rel_idx_start, rel_idx_end+1)
            try:
                fit_poly = np.polyfit(x_abcissa, x_fit, 2)
                rel_refined_max = -fit_poly[1]/fit_poly[0]/2
            except (ValueError, np.RankWarning):
                rel_refined_max = fit_points+1
            if np.abs(rel_refined_max) <= fit_points:
                t_max = (idx_max + rel_refined_max)/sr
                v_max = np.polyval(fit_poly, rel_refined_max)
            else:
                t_max = (idx_max)/sr
                v_max = x[idx_max]

        # prepare for next iteration 
        this_f0 = f[idx_max]
        if np.isfinite(this_f0):
            period_samp = int(sr/this_f0)
            marks.append(t_max)
            maxval.append(v_max)

        # otherwise keep the same period
        # next starting point
        min_search_max = np.min([idx_max+period_samp, len(x)])
        adv = np.argmin(x[idx_max:min_search_max])
        if adv > 0:
            idx_start = idx_max + adv
        else:
            idx_start = idx_max + 1
        
    return np.array(marks)[:-1], np.array(maxval)[:-1]

