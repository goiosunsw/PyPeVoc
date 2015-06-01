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
    def __init__(self, xw, sr=1, ncand = 8, candthresh = .8, vthresh = .2, mindelay=0, maxdelay=None, method='xcorr'):
        """Calculate the periodicity estimation for a window of a time signal
    
        Arguments: 
        x:          signal
        sr:         sample rate
        candthresh: ratio to lowest minima to keep as peak
        vthresh:    voicing threshold
        mindelay:   minimum value of period
        maxdelay:   maximum value of period
        ncand:      maximum number of period candidates
        method:     type of correlation correlation / matching to use
                    'xcorr' - correlation
                    'amdf'  - average mean difference function
                    'zc'    - zero crossing
        """

        nwind = len(xw)
        self.sr = sr
        
        self.mindelay = mindelay
        if maxdelay is None:
            self.maxdelay = round(nwind/2)
        else:
            self.maxdelay = maxdelay
            
        self.method = method
        self.threshold = candthresh
        self.vthresh = vthresh
        self.ncand = ncand
        
        self.cand_period = np.array([])
        self.cand_strength = np.array([])
        
        self._calc(xw)
        
        
    def _calc(self, xw):
        """Calculate the periodicity candidates
    
        Arguments: 
        xw: the windowed portion of time signal where periodicity is to be estimated
        """
        
        nwind = len(xw)
        
        # unvoiced
        pkpos = np.array([])
        pkstr = np.array([])

        
        try:
            if self.method is 'amdf':
                xc = amdf(xw)
            
                maxxc = max(xc)
                
                xcpos = (maxxc-xc[self.mindelay:self.maxdelay]) / maxxc
                xcth  = self.threshold
                
            elif self.method is 'xcorr':
                
                xc = np.correlate(xw,xw,"full") / self.wind
                xcred = xc[nwind-1+self.mindelay:nwind-1+self.maxdelay]
                xcpos = xcred/max(xc)
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
        
    def set_time_properties(self, index):
        """Set the sample and time value of this periodicity estimation
    
        Arguments: 
        index: sample index
        """

        self.index = index
        self.time = float(index)/self.sr
        
    def sort_strength(self):
        """Sort candidates by periodicity strength
    
        Arguments: (None)
        """

        idx = np.argsort(self.cand_strength)[::-1]
        self.cand_period = self.cand_period[idx]
        self.cand_strength = self.cand_strength[idx]







class PeriodTimeSeries(object):
    def __init__(self, x, sr=1, window=None, hop=None, threshold = .8, vthresh = .2, mindelay=0, maxdelay=None, ncand = 8, method = 'xcorr'):
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
        """
        
        self.method = method
        self.x = x.astype(float)
        self.sr = sr
        
        self.nx = len(x)
    
        if window is None:
            if maxdelay is None:
                window = self.nx
            else:
                window = 2*self.maxedlay
                
        if not np.iterable(window):
            window = np.ones(window)
        
        self.wind = window
        self.wnorm = self._calc_window_norm()
    
        self.nwind = len(window)
        self.windad = amdf(window)
        
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
 
        
        nwleft = np.floor(self.nwind/2)
        nwright = self.nwind - nwleft
        ist = index - nwleft
        iend = index + nwright
        
        xs = self.x[ist:iend]
        xw = xs * self.wind
        
        pp = Periodicity(xw, sr=self.sr, candthresh = self.threshold, vthresh = self.vthresh, mindelay = self.mindelay, maxdelay = self.maxdelay, ncand=self.ncand, method=self.method)
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
        
        
