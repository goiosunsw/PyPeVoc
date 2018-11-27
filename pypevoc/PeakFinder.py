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
        return self._val[self._keep]

    @property
    def all_val(self):
        return self._val

    @property
    def rough_val(self):
        return self._val

    def filter_by_salience(self, rad=1, sal=0):
        ''' Filters the peaks by salience.
            Any peak that is lower than the neighbouring 'rad' points
            is filtered out

            optional:
            * sal: salience (peaks must be at leas sal above other 
                   values in a radius rad)
        '''

        npks = len(self.pos)
        # keep = np.ones(npks).astype('bool')

        for idx in range(npks):
            thispos = self.pos[idx]
            thisval = self.val[idx]
            wmin = max(thispos-rad, 1)
            wmax = min(thispos + rad, len(self.y))
            w = self.y[wmin:wmax + 1]

            if any(w+sal > thisval):
                self.keep[idx] = False

        # self.keep = np.logical_and(self.keep, keep)

    def filter_by_prominence(self, prom=0.0, all=False):
        '''
        Filter by peak prominence

        prominence at leas prom above relative minimum

        optional:
        * all: include peaks that were filtered out before
        '''
        try:
            prominence = self.prominence
        except AttributeError:
            self.find_prominence()
            prominence = self.prominence

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

        self._idx = np.array(pos)
        self._val = np.array([y[i] for i in self._idx])
        self._keep = np.ones(len(self._idx)).astype('bool')
        self._fine_pos = self.x[self._idx]

    def find_prominence(self, side_fun=np.min, all=False):
        if not all:
            pos = self.pos
            val = self.val
        else:
            pos = self.all_pos
            val = self.all_val
        lbound = np.concatenate(([0], pos))
        rbound = np.concatenate((pos+1, [len(self.y)]))
        sal_l = []
        sal_r = []
        for lb, rb, val in zip(lbound[:-1], rbound[:-1], val):
            sal_l.append(val-np.min(self.y[lb:rb]))
        for lb, rb, val in zip(lbound[1:], rbound[1:], val):
            sal_r.append(val-np.min(self.y[lb:rb]))

        sal_l = np.array(sal_l)
        sal_r = np.array(sal_r)
        prominence = side_fun(np.array([sal_l,sal_r]),axis=0)

        if not all:
            self.prominence = prominence
        else:
            self.prominence = np.zeros(len(self.pos))
            self.prominence[self.keep] = prominence
        return self.prominence[self.keep]

    def plot(self, logarithmic=False):
        """Plot a graphical representation of the peaks

        Arguments:
            (none)
        """

        import pylab as pl

        pl.figure()
        pl.plot(self.x, self.y)
        #pl.hold('on')
        pl.plot(self.all_pos,
                self.all_val, 'om')
        pl.plot(self.rough_pos, self.rough_val, 'og')
        if hasattr(self, 'bounds'):
            lmins = np.unique(self.bounds.flatten())
            lminvals = self.y[lmins]
            pl.plot(lmins, lminvals, 'or')
        pl.plot(self.pos, self.val, 'dg')
        pl.hold('off')
        if logarithmic:
            pl.gca().set_yscale('log')

    def sort_ampl(self):
        """Sort the found peaks in decreasing order of amplitude

        Arguments:
            (none)
        """
        if len(self.pos) > 1:
            idx = np.argsort(self.val)[::-1]
            self.pos = self.pos[idx]
            self.val = self.val[idx]
            self.keep = self.keep[idx]
            self.sorttype = 2

    def sort_pos(self):
        """Sort the found peaks in order of position

        Arguments:
            (none)
        """

        if len(self.pos) > 1:
            idx = np.argsort(self.pos)

            self.pos = self.pos[idx]
            self.val = self.val[idx]
            self.keep = self.keep[idx]
            self.sorttype = 1

    def boundaries(self):
        """Find the local minima on either side of each peak

        Arguments:
            (none)
        """
        try:
            prevb = np.argmin(self.y[0:self.pos[0]])
        except IndexError:
            prevb = 0

        bounds = []

        npks = len(self.pos)

        if self.sorttype != 1:
            self.sort_pos()

        for i in range(npks):
            thismax = self.pos[i]
            if i < npks-1:
                nextmax = self.pos[i + 1]
                relb = np.argmin(self.y[thismax:nextmax])
                nextb = relb + thismax
            else:
                nextmax = len(self.y)-1
                nextb = len(self.y)-1

            bounds.append([prevb, nextb])
            prevb = nextb

        self.bounds = np.array(bounds)

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

        pos = self.pos[idx]
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

        return fpos, fval.tolist()

    def refine_all(self, logarithmic=False, rad=1):
        """use quadratic interpolation to refine all peaks

        Arguments:
            idx: index of the peak to interpolate
        """

        if logarithmic:
            y = np.log10(self.y)

        # rpos = self.pos
        # rval = self.val
        self.fpos = np.zeros(self.pos.shape)
        self.fval = np.zeros(self.pos.shape)

        for i in range(len(self.pos)):
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
            self.fpos[i] = fpos
            if logarithmic:
                self.fval[i] = 10**fval
            else:
                self.fval[i] = fval

    def get_pos(self, rough=False):
        """return a vector with peak position

        Arguments:
            rough: do not return the refined position
        """

        if hasattr(self, 'fpos') and not rough:
            return self.fpos[self.keep]
        else:
            return self.pos[self.keep]

    def get_val(self, rough=False):
        """return a vector with peak position

        Arguments:
            rough: do not return the refined position
        """

        if hasattr(self, 'fpos') and not rough:
            return self.fval[self.keep]
        else:
            return self.val[self.keep]

    def calc_individual_area(self, idx, funct=None, max_rad=None):
        lims = self.bounds[idx]
        if funct is None:
            return sum(self.y[lims[0]:lims[-1]])
        else:
            return sum(funct(self.y[lims[0]:lims[-1]]))

    def get_areas(self, funct=None, max_rad=None):
        if not hasattr(self, 'bounds'):
            self.boundaries()

        areas = []
        for idx in range(len(self.pos)):
            areas.append(self.calc_individual_area(idx, funct=funct))

        self.areas = np.array(areas)

        return self.areas[self.keep]

    def get_pos_val(self, rough=False):
        """return a vector with peak position in first column
        and value in second column

        Arguments:
            rough: do not return the refined position
        """

        if hasattr(self, 'fpos') and not rough:
            rvec = np.array(zip(self.fpos[self.keep], self.fval[self.keep]))
        else:
            rvec = np.array(zip(self.pos[self.keep], self.val[self.keep]))

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
                thisd['bounds'] = self.bounds[self._keep][ii]
            except AttributeError:
                pass
        
            ret.append(thisd)

        return ret
