#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PeakFinder.py
#
#  Provides a simple way of finding peaks and their parameters,
#  from a data series
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

    def __init__(self, y, x=None, npeaks=None, minrattomax=None, minval=None):
        """Creates the peak finder object from a numpy array

        Arguments:

            x:           the numpy array in which to find peaks
            npeaks:      maximum number of peaks to find

          Thresholds:
            minrattomax: ratio of minimum to maximum peak amplitude
                         (has priority over minval if set to other
                          than None)
            minval:      an absolute minimum value of peak
        """

        self.y = np.array(np.squeeze(y))
        if x is not None:
            self.x = np.array(np.squeeze(x))
        else:
            self.x = np.arange(len(self.y))
        self._idx = np.array([])
        self._val = np.array([])
        if minrattomax is None:
            self.minamp = minval
        else:
            self.minamp = self.y.max()*minrattomax

        self.sorttype = 0

        if not npeaks:
            self.npeaks = len(self.y)
        else:
            self.npeaks = npeaks

        if not self.minamp:
            self.minamp = np.min(self.y)

        self.findpos()
        #self.sort_pos()
        # self.boundaries()

    @property
    def pos(self):
        return self._fine_pos[self._keep]

    @property
    def rough_pos(self):
        return self.x[self._idx[self._keep]]

    @property
    def all_pos(self):
        return self.x[self._idx]

    @property
    def val(self):
        return self._fine_val[self._keep]

    @property
    def all_val(self):
        return self._val

    @property
    def rough_val(self):
        return self._val[self._keep]

    @property
    def bounds(self):
        b = np.array(self._bounds)
        return self.x[b[self._keep,:]]
    
    @property
    def areas(self):
        return self._areas[self._keep]

    @property
    def prominence(self):
        return self._prominence[self._keep]

    def filter_by_salience(self, rad=1, sal=0):
        ''' Filters the peaks by salience.
            Any peak that is lower than the neighbouring 'rad' points
            is filtered out

            optional:
            * sal: salience (peaks must be at leas sal above other 
                   values in a radius rad)
        '''

        npks = len(self._idx)
        # keep = np.ones(npks).astype('bool')

        for idx in range(npks):
            thispos = self._idx[idx]
            thisval = self._val[idx]
            wmin = max(thispos-rad, 1)
            wmax = min(thispos + rad, len(self.y))
            w = self.y[wmin:wmax + 1]

            if any(w+sal > thisval):
                self._keep[idx] = False

        # self.keep = np.logical_and(self.keep, keep)

    def filter_by_prominence(self, prom=0.0, all=False):
        '''
        Filter by peak prominence

        prominence at leas prom above relative minimum

        optional:
        * all: include peaks that were filtered out before
        '''
        try:
            prominence = self._prominence
        except AttributeError:
            self.find_prominence()
            prominence = self._prominence

        self._keep[prominence<prom] = False

    def findpos(self):
        """Finds the peaks positions

        Arguments:
            (none)
        """

        y = self.y

        miny = np.min(y)

        peakmask = (y[0:-2] < y[1:-1])*(y[1:-1] >= y[2:]).astype(int)
        pkmskamp = peakmask*(y[1:-1]-miny)
        # print(pkmskamp)

        pos = []

        m = pkmskamp.max()
        b = pkmskamp.argmax()
        th = self.minamp-miny
        n = 1

        if m > th:
            pos.append(b + 1)
            pkmskamp[b] = th-1

        while m > th and n < self.npeaks:
            m = pkmskamp.max()
            b = pkmskamp.argmax()
            if m > th:
                pos.append(b + 1)
                pkmskamp[b] = th-1
                n += 1

        self._idx = np.array(np.sort(pos))
        self._val = np.array([y[i] for i in self._idx])
        self._keep = np.ones(len(self._idx),dtype='bool')
        self._order = np.arange(len(self._idx))
        self._fine_pos = np.array([self.x[ii] for ii in self._idx])
        self._fine_val = self._val

    def find_prominence(self, side_fun=np.min, all=False):
        if not all:
            pos = self._idx[self._keep]
            val = self._val[self._keep]
        else:
            pos = self.all_pos
            val = self.all_val
        lbound = np.concatenate(([0], pos))
        rbound = np.concatenate((pos+1, [len(self.y)]))
        sal_l = []
        sal_r = []
        for lb, rb, v in zip(lbound[:-1], rbound[:-1], val):
            sal_l.append(v - np.min(self.y[lb:rb]))
        for lb, rb, v in zip(lbound[1:], rbound[1:], val):
            sal_r.append(v - np.min(self.y[lb:rb]))

        sal_l = np.array(sal_l)
        sal_r = np.array(sal_r)
        prominence = side_fun(np.array([sal_l,sal_r]),axis=0)

        if not all:
            self.prominence = prominence
        else:
            self.prominence = np.zeros(len(self.pos))
            self.prominence[self._keep] = prominence
        return self.prominence[self._keep]

    def plot(self, logarithmic=False):
        """Plot a graphical representation of the peaks

        Arguments:
            (none)
        """

        import pylab as pl

        pl.figure()
        pl.plot(self.x, self.y)
        pl.plot(self.all_pos,
                self.all_val, 'om')
        pl.plot(self.rough_pos, self.rough_val, 'og')
        if hasattr(self, 'bounds'):
            lmins = np.unique(self.bounds.flatten())
            lminvals = self.y[lmins]
            pl.plot(lmins, lminvals, 'or')
        pl.plot(self.pos, self.val, 'dg')
        if logarithmic:
            pl.gca().set_yscale('log')

    def sort_ampl(self):
        """Sort the found peaks in decreasing order of amplitude

        Arguments:
            (none)
        """
        if len(self.pos) > 1:
            idx = np.argsort(self._val)[::-1]
            self._order = idx
            self.sorttype = 2

    def sort_pos(self):
        """Sort the found peaks in order of position

        Arguments:
            (none)
        """

        if len(self._idx) > 1:
            idx = np.argsort(self._idx)

            self._order = idx
            self.sorttype = 1

    def find_boundaries(self, all=False):
        """Find the local minima on either side of each peak

        Arguments:
            (none)
        """
        try:
            prevb = np.argmin(self.y[0:self._idx[0]])
        except IndexError:
            prevb = 0

        bounds = []

        if not all:
            pos = self._idx[self._keep]
        else:
            pos = self._idx

        npks = len(pos)

        for i in range(npks):
            thismax = pos[i]
            if i < npks-1:
                nextmax = pos[i + 1]
                relb = np.argmin(self.y[thismax:nextmax])
                nextb = relb + thismax
            else:
                nextmax = len(self.y)-1
                nextb = len(self.y)-1

            bounds.append([prevb, nextb])
            prevb = nextb

        self._bounds = np.array(bounds)

    def refine_opt(self, idx, yvec=None, rad=2):
        """use fit to quadratic to locate a fine maximum of
        the peak position and value

        Arguments:
            idx: index of the peak to interpolate
        """

        pos = self.pos[idx]
        if yvec is not None:
            y = yvec
        else:
            y = self.y

        # val = self.val[idx]
        imin = max(1, pos-rad)
        imax = min(pos + rad + 1, len(y))
        sur = y[imin:imax]
        ifit = np.arange(imin-pos, imax-pos)

        pp = np.polyfit(ifit, sur, 2)
        lpos = - pp[1]/2.0/pp[0]
        fpos = float(pos) + lpos
        fval = pp[0]*lpos*lpos + pp[1]*lpos + pp[2]

        return fpos, fval.tolist()

    def refine(self, idx, fun=None, yvec=None):
        """use quadratic interpolation to locate a fine maximum of
        the peak position and value

        Arguments:
            idx: index of the peak to interpolate
        """

        pos = self._idx[idx]
        if yvec is not None:
            y = yvec
        else:
            y = self.y

        if fun:
            from scipy.optimize import broyden1 as opt
            # val = fun(self.val[idx])
            sur = fun(y[pos-1:pos+2])
        else:
            # val = self.val[idx]
            sur = y[pos-1:pos+2]

        if sur[1] > sur[0] and sur[1] >= sur[2]:
            c = sur[1]
            b = (sur[2] - sur[0])/2
            a = (sur[2] + sur[0])/2 - c

            lpos = - b/2/a
            fpos = float(pos) + lpos
            if fun:
                ival = a*lpos*lpos + b*lpos + c
                # print "rpos = %d; rf(val) = %f; f(val) = %f; dpos = %f;"%(pos, sur[1], ival, lpos)
                fval = opt(lambda x: fun(x)-ival, self.val[idx]/2)
            else:
                fval = a*lpos*lpos + b*lpos + c
                # print "rpos = %d; rval = %f; val = %f; dpos = %f; pos = %f"%(pos, sur[1], fval, lpos, fpos)

        else:
            fpos = pos
            fval = sur[1]

        return np.interp(fpos, np.arange(len(self.x)), self.x), fval.tolist()

    def refine_all(self, logarithmic=False, rad=1):
        """use quadratic interpolation to refine all peaks

        Arguments:
            idx: index of the peak to interpolate
        """

        if logarithmic:
            y = np.log10(self.y)
        else:
            y = self.y

        # rpos = self.pos
        # rval = self.val
        self._fine_pos = np.zeros(self._idx.shape)
        self._fine_val = np.zeros(self._idx.shape)

        for i in range(len(self._idx)):
            if logarithmic:
                if rad > 1:
                    fpos, fval = self.refine_opt(i, yvec=y, rad=rad)
                else:
                    fpos, fval = self.refine(i, yvec=y)
            else:
                if rad > 1:
                    fpos, fval = self.refine_opt(i, rad=rad)
                else:
                    fpos, fval = self.refine(i)
            self._fine_pos[i] = fpos
            if logarithmic:
                self._fine_val[i] = 10**fval
            else:
                self._fine_val[i] = fval

    def calc_individual_area(self, idx, funct=None, max_rad=None):
        lims = self._bounds[idx]
        if funct is None:
            return sum(self.y[lims[0]:lims[-1]])
        else:
            return sum(funct(self.y[lims[0]:lims[-1]]))

    def get_areas(self, funct=None, max_rad=None):
        if not hasattr(self, '_bounds'):
            self.find_boundaries()

        areas = []
        for idx in range(len(self._idx)):
            areas.append(self.calc_individual_area(idx, funct=funct))

        self._areas = np.array(areas)

        return self._areas[self._keep]

    def get_pos_val(self, rough=False):
        """return a vector with peak position in first column
        and value in second column

        Arguments:
            rough: do not return the refined position
        """

        rvec = np.array(zip(self.pos, self.val))

        return rvec

    def to_dict(self):
        """
        Return a list of dictionary with peak characteristics
        """
        ret = []
        for ii, (pos, val) in enumerate(zip(self.pos,self.val)):
            thisd = {'pos': pos,
                     'val': val}
            try:
                thisd['sal'] = self.prominence[self._keep][ii]
            except AttributeError:
                pass

            try:
                thisd['l_bound'] = self.bounds[ii,0]
                thisd['r_bound'] = self.bounds[ii,1]
            except AttributeError:
                pass

            try:
                thisd['area'] = self.areas[ii]
            except AttributeError:
                pass

            ret.append(thisd)

        return ret

    def to_data_frame(self):
        """
        Return a pandas dataframe with peak information
        """
        import pandas
        return pandas.DataFrame(self.to_dict())

    # backwards compat
    def boundaries(self):
        try:
            self._bounds
        except AttributeError:
            self.find_boundaries()
        b = np.array(self._bounds)
        try:
            return self.x[b[self._keep, :]]
        except IndexError:
            return np.array([])

    def get_pos(self):
        return np.array(self.pos)


