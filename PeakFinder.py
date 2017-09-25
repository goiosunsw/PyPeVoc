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

    def __init__(self, x, npeaks=None, minrattomax=None, minval=None):
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

        self.x = np.array(np.squeeze(x))
        self.pos = np.array([])
        self.val = np.array([])
        if minrattomax is None:
            self.minamp = minval
        else:
            self.minamp = self.x.max()*minrattomax

        self.sorttype = 0

        if not npeaks:
            self.npeaks = len(self.x)
        else:
            self.npeaks = npeaks

        if not self.minamp:
            self.minamp = np.min(self.x)

        self.findpos()
        self.sort_pos()
        # self.boundaries()

    def filter_by_salience(self, rad=1):
        ''' Filters the peaks by salience.
            Any peak that is lower than the neighbouring 'rad' points
            is filtered out
        '''

        npks = len(self.pos)
        # keep = np.ones(npks).astype('bool')

        for idx in range(npks):
            thispos = self.pos[idx]
            thisval = self.val[idx]
            wmin = max(thispos-rad, 1)
            wmax = min(thispos + rad, len(self.x))
            w = self.x[wmin:wmax + 1]

            if any(w > thisval):
                self.keep[idx] = False

        # self.keep = np.logical_and(self.keep, keep)

    def findpos(self):
        """Finds the peaks positions

        Arguments:
            (none)
        """

        x = self.x

        minx = np.min(x)

        peakmask = (x[0:-2] < x[1:-1])*(x[1:-1] >= x[2:]).astype(int)
        pkmskamp = peakmask*(x[1:-1]-minx)
        # print(pkmskamp)

        pos = []

        m = pkmskamp.max()
        b = pkmskamp.argmax()
        th = self.minamp-minx
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

        self.pos = np.array(pos)
        self.val = np.array([x[i] for i in self.pos])
        self.keep = np.ones(len(self.pos)).astype('bool')

    def plot(self, logarithmic=False):
        """Plot a graphical representation of the peaks

        Arguments:
            (none)
        """

        import pylab as pl

        pl.figure()
        pl.plot(self.x)
        pl.hold('on')
        pl.plot(self.pos[self.keep], self.val[self.keep], 'og')
        pl.plot(self.pos[np.logical_not(self.keep)],
                self.val[np.logical_not(self.keep)], 'om')
        if hasattr(self, 'bounds'):
            lmins = np.unique(self.bounds.flatten())
            lminvals = self.x[lmins]
            pl.plot(lmins, lminvals, 'or')
        if hasattr(self, 'fpos'):
            pl.plot(self.fpos[self.keep], self.fval[self.keep], 'dg')
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
            prevb = np.argmin(self.x[0:self.pos[0]])
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
                relb = np.argmin(self.x[thismax:nextmax])
                nextb = relb + thismax
            else:
                nextmax = len(self.x)-1
                nextb = len(self.x)-1

            bounds.append([prevb, nextb])
            prevb = nextb

        self.bounds = np.array(bounds)

    def refine_opt(self, idx, xvec=None, rad=2):
        """use fit to quadratic to locate a fine maximum of
        the peak position and value

        Arguments:
            idx: index of the peak to interpolate
        """

        pos = self.pos[idx]
        if xvec is not None:
            x = xvec
        else:
            x = self.x

        # val = self.val[idx]
        imin = max(1, pos-rad)
        imax = min(pos + rad + 1, len(x))
        sur = x[imin:imax]
        ifit = np.arange(imin-pos, imax-pos)

        pp = np.polyfit(ifit, sur, 2)
        lpos = - pp[1]/2.0/pp[0]
        fpos = float(pos) + lpos
        fval = pp[0]*lpos*lpos + pp[1]*lpos + pp[2]

        return fpos, fval.tolist()

    def refine(self, idx, fun=None, xvec=None):
        """use quadratic interpolation to locate a fine maximum of
        the peak position and value

        Arguments:
            idx: index of the peak to interpolate
        """

        pos = self.pos[idx]
        if xvec is not None:
            x = xvec
        else:
            x = self.x

        if fun:
            from scipy.optimize import broyden1 as opt
            # val = fun(self.val[idx])
            sur = fun(x[pos-1:pos+2])
        else:
            # val = self.val[idx]
            sur = x[pos-1:pos+2]

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
            x = np.log10(self.x)

        # rpos = self.pos
        # rval = self.val
        self.fpos = np.zeros(self.pos.shape)
        self.fval = np.zeros(self.pos.shape)

        for i in range(len(self.pos)):
            if logarithmic:
                if rad > 1:
                    fpos, fval = self.refine_opt(i, xvec=x, rad=rad)
                else:
                    fpos, fval = self.refine(i, xvec=x)
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
            return sum(self.x[lims[0]:lims[-1]])
        else:
            return sum(funct(self.x[lims[0]:lims[-1]]))

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
