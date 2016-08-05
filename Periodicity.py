#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  AMDF.py
#  
#  Utilities based on the Average Maen Difference Function
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
import PeakFinder as pf
import pylab as pl
from matplotlib.colors import hsv_to_rgb


def amdf(x, mindelay = 0, maxdelay = None):
    nx = len(x)
    if maxdelay is None:
        maxdelay = nx
        
    y = np.zeros(nx)
    for i in range(mindelay,maxdelay):
        n = nx - i
        y[i] = (np.abs(x[0:nx-i]-x[i:])).sum()/n
    
    return y

# I will try to update this object so that data required for the initialisation of every instance stays in the caller. Thee caller passes itself as argument to the callee

class Periodicity(object):
    """Single period object, including multiple periodicity candidates
    """
    def __init__(self, parent, index=0):
        """Calculate the periodicity estimation for a window of a time signal
    
        Arguments: 
        parent: parent object contaigning entire signal
        idx:    index of local peridoicity calulation
        """

        self.nwind = parent.nwind
        self.wnorm = parent.wnorm
        self.wind = parent.wind
        self.sr = parent.sr
        
        self.mindelay = parent.mindelay
        if parent.maxdelay is None:
            self.maxdelay = round(self.nwind/2)
        else:
            self.maxdelay = parent.maxdelay
            
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
        
        nwleft = np.floor(self.nwind/2)
        nwright = self.nwind - nwleft
        idx=int(np.round(index))
        ist = idx - nwleft
        iend = idx + nwright
        
        xs = parent.x[ist:iend]
        xw = xs * self.wind
        self.cand_method = parent.cand_method

        
        self._calc(xw)
        self.parent = parent
        
        
    def _calc(self, xw):
        """Calculate the periodicity candidates
    
        Arguments: 
        xw: the windowed portion of time signal where periodicity is to be estimated
        """
        
        nwind = self.nwind
        
        # unvoiced
        pkpos = np.array([])
        pkstr = np.array([])
        
        peaks = None
        
        try:
            if self.method is 'amdf':
                xc = amdf(xw)
            
                maxxc = max(xc)
                xcn = (maxxc-xc)/maxxc
                
                xcpos = xcn[self.mindelay:self.maxdelay]
                xcth  = self.threshold
                
            elif self.method is 'xcorr':
                
                xc = np.correlate(xw,xw,"full") / self.wnorm
                xcn = xc/max(xc)
                xcpos = xcn[nwind-1+self.mindelay:nwind-1+self.maxdelay]

                xcth = self.threshold
                
                #print "In xcorr. max %f, thr %f"%(max(xcpos),xcth)
                
            if max(xcpos) > self.vthresh:
                # this is equivlent to finding minima below the absolute minimum * threshold
                peaks = pf.PeakFinder(xcpos, minval = xcth, npeaks = self.ncand)
                
                
                peaks.refine_all()
                #peaks.plot()
                
                pkpos = peaks.get_pos() + self.mindelay
                pkstr = peaks.get_val()
                
                #keep = pkpos<self.maxdelay
                #pkpos = pkpos[keep]
                #pkstr = pkstr[keep]
                

            
        except IndexError:
            pass
            
        self.cand_period = pkpos
        self.cand_strength = pkstr
        
        if self.cand_method == 'fft':
            xf = np.fft.fft(xw)
            fftpeaks = pf.PeakFinder(np.abs(xf[0:self.nwind/2]), npeaks = self.ncand)
            # periodicity corresponding to fft peaks:
            fpos = fftpeaks.get_pos()
            fval = fftpeaks.get_val()
            fposkeep = fpos[fval>np.max(fval*self.fftthresh)]
            fftpkpos = self.nwind / fposkeep
            
            # minimum distance between correlation candidates and fft peaks
            perdist = [np.min(np.abs(fftpkpos-thispos)) for thispos in pkpos]
            try:
                self.preferred = np.argmin(perdist)
            except ValueError:
                self.preferred=0
            #print (fftpkpos)
            #print (pkpos)
        elif self.cand_method == 'min':
            self.preferred = np.argmin(pkpos)
        elif self.cand_method == 'similar':
            self.preferred = np.argmax(pkstr)

        
        return xcn
        
    def plot_similarity(self):
        
        nwleft = np.floor(self.nwind/2)
        nwright = self.nwind - nwleft
        ist = self.index - nwleft
        iend = self.index + nwright
        
        xs = self.parent.x[ist:iend]
        xw = xs * self.wind
        
        xc = self._calc(xw)
        pl.figure()
        pl.plot(np.arange(len(xc))-self.nwind,xc)
        pl.hold('on')
        pl.plot(self.cand_period, self.cand_strength, 'o')
            
            
            
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
        self.preferred = np.flatnonzero(idx==self.preferred)
    
    def get_preferred_period(self):
        if len(self.cand_period)>0:
            return self.cand_period[self.preferred]
        else:
            return self.mindelay





class PeriodTimeSeries(object):
    def __init__(self, x, sr=1, window=None, hop=None, 
                 threshold = .8, vthresh = .2, 
                 mindelay=1, maxdelay=None, 
                 ncand = 8, method = 'xcorr',
                 cand_method='fft', fftthresh=0.1):
        """Calculate the average mean difference of x around index
    
        Arguments: 
        x:         signal
        sr:        sample rate
        window:    window around index used for difference calculations
        threshold: ratio to lowest minima to keep as peak
        vthresh:   voicing threshold
        mindelay:  minimum value of period
        maxdelay:  maximum value of period
        ncand:     maximum number of period candidates
        method:    type of correlation correlation / matching to use
                   'xcorr' - correlation
                   'amdf'  - average mean difference function
                   'zc'    - zero crossing
        cand_method: method for candidate estimation:
                     'fft'    - based on an fft of the window
                     'min'    - minimum periodicity wins
                     'similar'- most similar wins
        fftthresh: threshold for fft peak selection (default=0.1)
        """
        
        self.method = method
        self.x = x.astype(float)
        self.sr = sr
        
        self.nx = len(x)
    
        if window is None:
            if maxdelay is None:
                window = self.nx
            else:
                window = 2*self.maxdelay
                
        if not np.iterable(window):
            window = np.ones(window)
        
        self.wind = window
        self._calc_window_norm()
    
        self.nwind = len(window)
        #self.windad = amdf(window)
        
        if hop is None:
            hop = round(self.nwind/2)
        
        self.hop = hop
        
        self.mindelay = mindelay
        if maxdelay is None:
            self.maxdelay = round(self.nwind/2)
            
        self.method = method
        self.threshold = threshold
        self.vthresh = vthresh
        self.ncand = ncand
        self.cand_method = cand_method
        self.fftthresh=fftthresh
        
        # data storage
        self.periods = []
        
    def _calc_window_norm(self):
        """Calculate the normalisation function for window
    
        Arguments: (None)
        """
        
        if self.method is 'xcorr':
            w = self.wind
            self.wnorm = np.correlate(w,w,"full")
        else:
            self.wnorm = 1.   


    def per_at_index(self,index):
        """Calculate the average mean difference of x around index
    
        Arguments: 
    
        index:  index of x for current amdf
        threshold: ratio to lowest minima to keep as peak
        """
 
        
        
        pp = Periodicity(self, index)
        pp.set_time_properties(index)
        pp.sort_strength()

        self.periods.append(pp)
            
            
    def calc(self, hop=None, threshold=None):
        """Estimate local periodicity in the full time series
    
        Arguments: 
    
        hop:       samples bewteen estimations
        threshold: peak threshold for maintaining or rejecting 
                   candidates
        """

        
        if hop is None:
            hop=self.hop
        
        if threshold is not None:
            oldthresh = self.threshold
            self.threshold = threshold
        
        idxmax = self.nx-self.nwind
        idxvec = np.arange(self.nwind,idxmax,hop)
        
        sys.stdout.write("Calculating local periodicity... "  ) 
        
        for idx in idxvec:
            self.per_at_index(idx)
            sys.stdout.write("\b\b\b\b%3d%%" % (idx*100/idxmax) )
            sys.stdout.flush()

        sys.stdout.write("\ndone\n"  ) 
        
        if threshold is not None:
            self.threshold = oldthresh
  
    def calcPeriodByPeriod(self, threshold=None):
        """Estimate local periodicity in the full time series
    
        Arguments: 
    
        hop:       samples bewteen estimations
        threshold: peak threshold for maintaining or rejecting 
                   candidates
        """
        
        if threshold is not None:
            oldthresh = self.threshold
            self.threshold = threshold
        
        # Max index for starting window
        idxmax = self.nx-self.nwind
        
        sys.stdout.write("Calculating local periodicity... "  ) 
        idx=self.nwind
        while idx < idxmax:
            self.per_at_index(idx)
            per_obj = self.periods[-1]
            idx += per_obj.get_preferred_period() 
            sys.stdout.write("\b"*15+"%6d / %6d" % (idx,self.nx) )
            sys.stdout.flush()

        sys.stdout.write("\ndone\n"  ) 
        
        if threshold is not None:
            self.threshold = oldthresh
 
    def plot_candidates(self):
        """Plot a representation of candidate periodicity
        
        Size gives the periodicity strength, color the order of preference
        """
        
        hues = np.arange(self.ncand)/float(self.ncand)
        hsv = np.swapaxes(np.atleast_3d([[hues,np.ones(len(hues)),np.ones(len(hues))]]),1,2)
        cols = hsv_to_rgb(hsv).squeeze()
        
        for per in self.periods:
            nc = len(per.cand_period)
            
            pl.scatter(per.time*np.ones(nc),per.cand_period,s=per.cand_strength*100,c=cols[0:nc],alpha=.5)
        
        pl.plot(zip([[per.time,per.get_preferred_period] for per in self.periods]),'k')
        
    def get_f0(self):
        """Get f0 as a function of time
        """
        
        f0 = np.zeros(len(self.periods))
        for ii,per in enumerate(self.periods):
            if len(per)>0:
                f0[ii]=self.sr/min(per)
        return f0
