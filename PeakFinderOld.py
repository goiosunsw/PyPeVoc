#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PeakFinder.py
#  
#  Copyright 2014 Andre Almeida <andre.almeida@univ-lemans.fr>
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

""" Defines a class for detecting peaks in a numpy array"""

import numpy as np

class PeakFinder(object):

    def __init__(self, x, npeaks=None, threshold=0):
        """Creates the peak finder object from a numpy array
        
        Arguments: 
        
            x:         the numpy array in which to find peaks
            npeaks:    maximum number of peaks to find
            threshold: ratio of minimum to maximum peak amplitude
        """
        
        
        
        self.x = np.array(x)
        self.peaks()
        if not npeaks:
            self.npeaks = length(self.pos)
            
        self.apply_threshold(threshold)
        
        self.sort_ampl()
        
        
        
    def peaks(self):
        """Finds the peaks
        
        Arguments:
            (none)
        """
        
        x=self.x
        
        peakmask = np.logical_and(x[0:-2]<x[1:-1],x[1:-1]>x[2:])
    
        self.pos = np.nonzero(peakmask)[0]+1
        self.val = x[self.pos]
        
    def plot(self):
        """Plot a graphical representation of the peaks
        
        Arguments:
            (none)
        """
    
        import pylab as pl
        
        pl.figure()
        pl.plot(self.x)
        pl.hold('on')
        pl.plot(self.pos,self.val,'o')
        pl.hold('off')
        
    def sort_ampl(self):
        """Sort the found peaks in decreasing order
        
        Arguments:
            (none)
        """
        
        idx = np.argsort(self.val)[::-1]
        self.pos = self.pos[idx]
        self.val = self.val[idx]
        
    def apply_threshold(threshold=0):
        """Filter peaks according to a threshold ratio relative to max 
        amplitude
        
        Arguments:
            threshold: minimum ratio to max peak amplitude
        """
        
        val = self.val
        max_amp = val.max()
        min_amp = max_amp * threshold
        
        for i = range(self.npeaks):
            
