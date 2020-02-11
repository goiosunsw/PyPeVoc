#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PVanalyse.py
#
#  Do a Phase vocoder decomposition of a signal into its quasi-sinusoidal
# components
#
#  Copyright 2015 Andre Almeida <andregoios@gmail.com>
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

import numpy as np
import pylab as pl
import sys

from .PeakFinder import PeakFinder as pf
from .ProgressDisplay import Progress

try:
    from scipy.interpolate import interp1d
except ImportError:
    sys.stderr.write('SciPy not found. Using linear interpolation\n')

"""
Perform the Phase vocoder decomposition of a signal into
quasi-sinusoidal components
"""

pi2 = 2.0*np.pi

# approximate formulas for logarithmic ratios

# 20*log10(rat) approx= dbconst*(rat-1)
dbconst = 20./np.log(10)
# 12*log2(rat) approx= dfconst*(rat - 1)
dfconst = 12./np.log(2)


def dpitch2st_exact(f1, f2):
    '''
    Convert a frequency interval to semitones
    '''
    return 12*np.log2(float(f2)/f1)


def dpitch2st(f1, f2):
    '''
    Convert a frequency interval between f1 and f2 to semitones
    (aprroximate formula for nearby freq.
     for exact formula use dpitch2st_exact(f1, f2))
    '''
    return 17.312*(float(f2)/f1 - 1.0)


class PV:
    def __init__(self, x, sr, nfft=1024, hop=None, npks=20,
                 pkthresh=0.005, wind=np.hanning, progress=True):
        '''
        Phase vocoder object.
        Arguments:
            * sr   = Sampling rate
            * nfft = Number of points in FFT analysis window
            * hop  = Number of points between FFT windows
            * npks = Maximum number of peaks at each frame
            * pkthresh = Threshold of peak amplitude relative of maximum
        '''

        self.x = np.array(x)
        self.nsamp = len(self.x)
        self.sr = sr
        self.nfft = nfft
        self.nfft2 = int(nfft/2)
        if hop is None:
            self.hop = int(self.nfft/2)
        else:
            self.hop = hop
        self.peakthresh = pkthresh
        self.npeaks = npks
        self.nframes = 0

        self.win = wind(nfft)
        self.wsum = sum(self.win)
        self.wsum2 = sum(self.win**2)
        # self.wfact = self.wsum#*np.sqrt(self.nfft);
        # I don't remember why this is the normalisation factor...
        self.wfact = np.sqrt(self.wsum2*self.nfft)/2.0

        # the freqency step between bins
        self.fstep = float(self.sr)/float(self.nfft)

        # time difference beween frames
        self.dt = float(self.hop)/float(self.sr)

        # multiples of 2pi
        # self.all2pi = 2*np.pi*np.arange(0:round(self.hop/2.0))

        # central freq of each bin
        self.fbin = np.arange(float(nfft))*self.fstep
        # pase difference between frames for each bin
        dthetabin = pi2*self.fbin*self.dt
        # wrapping factor for each bin * 2pi
        self.wfbin = np.round(dthetabin/pi2) * pi2

        # storage for the older fft frame
        self.oldfft = np.zeros(self.nfft2)

        # calculated values
        self.t = []
        self.f = []
        self.ph = []
        self.mag = []
        if progress:
            self.progress = Progress(end=self.nsamp)
        else:
            self.progress = None

    def dphase2freq(self, dph, nbin):
        '''
        Calculates the "instantaneous frequency" corresponding to the
        phase difference dph between two consecutive frames
        '''
        # Unwrapped phase
        # dphw = dph + self.wfbin[nbin] + np.array([-pi2, 0, pi2])
        dphw = dph + self.wfbin[nbin] + pi2*np.arange(-1, 2)
        # precise frequency options
        freq = dphw / self.dt / pi2
        # search among neighboring bins for the right freq
        df = self.fbin[nbin] - freq
        ii = np.argmin(abs(df))

        return freq[ii], df[ii]
        # return self.fbin[nbin]

    def calc_fft_frame(self, pos):
        '''
        Calculate a FFT frame at pos
        '''

        thisx = self.x[pos:pos+self.nfft]
        xw = thisx*self.win
        fx = np.fft.fft(xw) / self.wfact
        return fx

    def calc_pv_frame(self, pos):
        '''
        Determine PV peaks and calculate frequencies
        based on previous fft frame
        '''

        wd = 1

        fxa = self.calc_fft_frame(pos)
        fx = fxa[:self.nfft2]

        frat = fx / self.oldfft

        famp = abs(fx)
        # find the peaks in the FFT
        pkf = pf(famp, npeaks=self.npeaks, minrattomax=self.peakthresh)
        pkf.boundaries()
        pkf.filter_by_salience(rad=5)
        pk = pkf.get_pos()

        f = []
        mag = []
        ph = []
        realph = []
        binno = []

        # for each peak
        for ipk, nbin in enumerate(pk):
            thisph = np.angle(fx[nbin])
            # pahse difference
            dph = np.angle(frat[nbin])
            freq, df = self.dphase2freq(dph, nbin)

            if freq > 0.0:
                binno.append(nbin)
                f.append(freq)
                # amplitude
                imin = max(nbin - wd, 1)
                imax = min(nbin + wd, len(famp))
                mag.append(np.sqrt(sum(famp[imin:imax+1]**2)))
                # mag.append(np.sqrt(pkf.calc_individual_area(ipk,
                # funct=lambda x:x*x)))

                ph.append(thisph)
                # phase correction for varying frequency
                # phcor = np.pi*(fsam[-1] - fsam[0])/fstep/2.;

                realph.append(thisph + np.pi * df/self.fstep)

        self.oldfft = fx
        totalmag = np.sqrt(np.sum(famp**2))
        return f, mag, ph, realph, binno, totalmag

    def run_pv(self):

        allf = []
        allmag = []
        allph = []
        allrealph = []
        allbin = []
        totalmag = []
        t = []

        curpos = 0
        maxpos = self.nsamp - self.nfft
        while curpos < maxpos:
            f = np.zeros(self.npeaks)
            mag = np.zeros(self.npeaks)
            ph = np.zeros(self.npeaks)
            realph = np.zeros(self.npeaks)
            binno = np.zeros(self.npeaks)

            ff, magf, phf, realf, binf, tmag = self.calc_pv_frame(curpos)
            totalmag.append(tmag)

            f[0:len(ff)] = ff
            mag[0:len(magf)] = magf
            ph[0:len(phf)] = phf
            realph[0:len(realf)] = realf
            binno[0:len(realf)] = binf

            allf.append(f)
            allmag.append(mag)
            allph.append(ph)
            allrealph.append(realph)
            allbin.append(binno)

            t.append((curpos + self.nfft/2.0)/self.sr)

            curpos += self.hop
            if self.progress:
                self.progress.update(curpos)

        if self.progress:
            self.progress.update(self.nsamp)

        self.f = np.array(allf)
        self.mag = np.array(allmag)
        self.ph = np.array(allph)
        self.realph = np.array(allrealph)
        self.binno = np.array(allbin)
        # time values
        self.t = np.array(t)
        self.nframes = len(t)
        self.totalmag = totalmag

    def calc_harmonic_power(self, f_threshold=0.01):
        """
        calculate the harmonic power of individual sine components
        """

        hpower = []
        nharmonics = []

        for nfr in range(self.f.shape[0]):
            this_f = self.f[nfr, :]
            valid_idx = np.flatnonzero(this_f > 0)
            valid_f = this_f[valid_idx]
            valid_mag = self.mag[valid_idx]
            valid_hpower = []
            valid_n_harm = []
            for f, mag in zip(valid_f, valid_mag):
                harmonic_nbr = np.round(valid_f/f)
                harmonic_nbr[harmonic_nbr == 0] = 1
                inharmonicity = np.abs(valid_f/harmonic_nbr/f - 1)
                harmonic_comp = np.flatnonzero(inharmonicity < f_threshold)
                valid_hpower.append(np.sum(valid_mag[harmonic_comp]**2))
                valid_n_harm.append(len(harmonic_comp))

            this_hpower = np.zeros(len(this_f))
            this_hpower[valid_idx] = valid_hpower
            hpower.append(this_hpower)
            this_n_harm = np.zeros(len(this_f))
            this_n_harm[valid_idx] = valid_n_harm
            nharmonics.append(this_n_harm)

        self.hpower = np.array(hpower)
        self.nharmonics = np.array(nharmonics)

    def toSinSum(self, maxpitchjmp=0.5):
        '''
        Convert to Sine sum
        Arguments:
            * maxpitchjmp = maximum allowed jump in pitch between frames
                            (in semitones)
        '''
        ss = SinSum(self.sr, nfft=self.nfft, hop=self.hop)

        # lastf     = np.zeros(self.npeaks)
        # lastssidx = np.nan*np.zeros(self.npeaks)

        for fr in range(self.nframes):
            # process new peaks in decreasing magnitude
            # irev = np.argsort(self.mag[fr, :])
            # idx = irev[::-1]
            # ffr = self.f[fr, idx]
            # mfr = self.mag[fr, idx]
            # pfr = self.ph[fr, idx]
            # for f, mag, ph in zip(ffr, mfr, pfr):
            # ss.add_point(fr, f, mag, ph, maxpitchjmp=maxpitchjmp)
            ss.add_frame(fr, self.f[fr, :], self.mag[fr, :], self.ph[fr, :],
                         realph=self.realph[fr, :])
        return ss

    def plot_time_freq(self, colors=True, ax=None):
        import pylab as pl

        if ax is None:
            fig, allax = pl.subplots(1)
            ax = allax

        # make time matrix same shape as others
        t = np.outer(self.t, np.ones(self.npeaks))
        f = self.f
        if colors:
            mag = 20*np.log10(self.mag)
            ax.scatter(t, f, s=6, c=mag, lw=0)
        else:
            mag = 100 + 20*np.log10(self.mag)
            ax.scatter(t, f, s=mag, lw=0)
        pl.xlabel('Time (s)')
        pl.ylabel('Frequency (Hz)')
        # if colors:
        # cs = pl.colorbar(ax=ax)
        # cs.set_label('Magnitude (dB)')
        # pl.show()
        return ax

    def plot_time_mag(self):
        import pylab as pl

        pl.figure()
        t = np.outer(self.t, np.ones(self.npeaks))
        # f = np.log2(self.f)
        f = self.f
        mag = 20*np.log10(self.mag)
        pl.scatter(t, mag, s=10, c=f, lw=0,
                   norm=pl.matplotlib.colors.LogNorm())
        pl.xlabel('Time (s)')
        pl.ylabel('Magnitude (dB)')
        cs = pl.colorbar()
        cs.set_label('Frequency (Hz)')
        # pl.show()
        return pl.gca()

    def get_time_vector(self):
        return self.t

    def get_sample_vector(self):
        return (self.t*self.sr).astype('int')

    def calc_f0(self, fmin=50, fmax=10000, thr=0.1):
        """
        Determine fundamental components in periodic tones
        and return their freuqnecy
        """
        fm = np.zeros(self.f.shape[0])
        im = np.zeros(self.f.shape[0],dtype='i')
        for ii in range(len(fm)):
            ff = self.f[ii,:]
            mm = self.mag[ii,:]
            maxmag = np.max(mm)
            in0 = np.flatnonzero(np.all((ff>fmin,ff<fmax,mm>maxmag*thr),axis=0))
            if len(in0)>0:
                fn0 = ff[in0]
                #isel = np.argmax(mm[in0])
                isel = np.argmin(ff[in0])
                fm[ii] = fn0[isel]
                im[ii] = in0[isel]

        self.fundamental_idx = im
        return fm

    @property
    def fundamental_frequency(self):
        try:
            return self.f[np.arange(self.f.shape[0]),
                          self.fundamental_idx]
        except AttributeError:
            return self.calc_f0()

    @property
    def fundamental_magnitude(self):
        try:
            return self.mag[np.arange(self.f.shape[0]),
                            self.fundamental_idx]
        except AttributeError:
            self.calc_f0()
            return self.mag[np.arange(self.f.shape[0]),
                            self.fundamental_idx]

    @property
    def partial_sum_magnitude(self):
        return np.sqrt(np.sum(self.mag**2,axis=1))

    @property
    def partial_magnitude_ratio(self):
        return self.partial_sum_magnitude/self.totalmag

class PVHarmonic(PV):
    def __init__(self, *args, **kwargs):
        self.fmin = 30.0
        PV.__init__(self, *args, **kwargs)

    def set_f0(self, f0, t=None):
        '''
        Assign a f0 vector to the search
        Argument:
            * f0: f0 vector over time
            * t: if present, values of time corresponding to f0
                 otherwise, the tie values correspond to the hop size
        '''

        # internal time vector
        tint = np.arange(round(self.hop + self.nfft/2),
                         len(self.x), self.hop)/float(self.sr)

        if t is None:
            self.f0 = f0
        else:
            self.f0 = np.interp(tint, t, f0)

    def calc_pv_frame(self, pos, f0):
        '''
        Determine PV peaks and calculate frequencies
        based on previous fft frame
        '''

        wd = 1

        fxa = self.calc_fft_frame(pos)
        fx = fxa[:self.nfft2]

        frat = fx / self.oldfft

        famp = abs(fx)

        f = []
        mag = []
        ph = []
        cummagsq = 0

        # for each f0 multiple
        f0bin = f0/self.sr*self.nfft
        bins = np.round(np.arange(f0bin, self.nfft2 - 1, f0bin)).astype('int')
        for ipk, nbin in enumerate(bins):
            if ipk > 0:
                if f[0] > self.fmin:
                    corrbin = f[0]/self.sr*self.nfft*(ipk + 1)
                    if corrbin < self.nfft2 - 1:
                        nbin = int(round(corrbin))
                        # print nbin

            thisph = np.angle(fx[nbin])
            # pahse difference
            dph = np.angle(frat[nbin])
            freq, df = self.dphase2freq(dph, nbin)
            # freq = ipk*f0

            f.append(freq)
            # amplitude
            imin = max(nbin - wd, 1)
            imax = min(nbin + wd, len(famp))
            thismagsq = (sum(famp[imin:imax+1]**2))
            cummagsq += thismagsq
            mag.append(np.sqrt(thismagsq))
            # mag.append(np.sqrt(pkf.calc_individual_area(ipk,
            # funct=lambda x:x*x)))

            ph.append(thisph)
        residual = np.sqrt(np.sum(famp**2)-cummagsq)
        self.oldfft = fx
        return f, mag, ph, residual

    def run_pv(self):

        allf = []
        allmag = []
        allph = []
        allres = []
        t = []

        curpos = 0
        maxpos = self.nsamp - self.nfft
        while curpos < maxpos:
            f = np.zeros(self.npeaks)
            mag = np.zeros(self.npeaks)
            ph = np.zeros(self.npeaks)
            thisf = self.f0[int((curpos)/self.hop)]
            residual = np.nan

            if thisf>0 and ~np.isnan(thisf):
                ff, magf, phf, residual = self.calc_pv_frame(curpos, thisf)

                nh = min(len(ff), len(f))

                f[0:nh] = ff[0:nh]
                mag[0:nh] = magf[0:nh]
                ph[0:nh] = phf[0:nh]

            allf.append(f)
            allmag.append(mag)
            allph.append(ph)
            allres.append(residual)

            t.append((curpos + self.nfft/2.0)/self.sr)

            curpos += self.hop
            self.progress.update(curpos)

        self.progress.update(self.nsamp)

        self.f = np.array(allf)
        self.mag = np.array(allmag)
        self.ph = np.array(allph)
        self.residuals = np.array(allres)
        # time values
        self.t = np.array(t)
        self.nframes = len(t)


class Partial(object):
    def __init__(self, pdict=None):
        '''
        A quasi-sinusoidal partial
        Arguments: pdict with:
            * pdict['t'] = time array
            * pdict['f'] = frequency array
            * pdict['mag'] = magnitude array
            * pdict['ph'] = phase array
        '''

        if pdict is None:
            self.t = []
            self.f = []
            self.mag = []
            self.ph = []
        else:
            self.t = pdict['t']
            self.f = pdict['f']
            self.mag = pdict['mag']
            self.ph = pdict['ph']

    def add_point(self, t, f, mag, ph):
        '''
        Append a single point in the partial
        '''
        if t > max(self.t):
            self.t.append(t)
            self.f.append(f)
            self.mag.append(mag)
            self.ph.append(ph)
        else:
            idx = (self.t > t).index(True)
            self.t.insert(idx, t)
            self.f.insert(idx, f)
            self.mag.insert(idx, mag)
            self.ph.insert(idx, ph)

    def synth(self, sr):
        '''
        Resynthesise the sinusoidal partial at sampling rate sr
        '''


class RegPartial(object):
    def __init__(self, istart, pdict=None, overlap=0.5, fstep=None):
        '''
        A quasi-sinusoidal partial with homogeneous sampling
        Arguments:
          istart = starting index
          pdict with:
            * pdict['t'] = time array
            * pdict['f'] = frequency array
            * pdict['mag'] = magnitude array
            * pdict['ph'] = phase array
        '''

        self.start_idx = istart
        self.overlap = overlap
        self.fstep = fstep

        if pdict is None:
            self.f = []
            self.mag = []
            self.ph = []
            self.realph = []
        else:
            self.f = pdict['f']
            self.mag = pdict['mag']
            self.ph = pdict['ph']
            try:
                self.realph = pdict['realph']
            except KeyError:
                self.realph = pdict['ph']

    def append_point(self, f, mag, ph, realph=None):
        '''
        Add a single point to the end of partial
        '''
        self.f.append(f)
        self.mag.append(mag)
        self.ph.append(ph)
        if realph is None:
            self.realph.append(ph)
        else:
            self.realph.append(realph)

    def prepend_point(self, f, mag, ph):
        '''
        Add a single point to the start of partial
        '''
        self.f.insert(0, f)
        self.mag.insert(0, mag)
        self.ph.insert(0, ph)
        self.start_idx -= 1

    def get_freq_at_frame(self, fr):
        relidx = fr - self.start_idx

        if relidx >= 0:
            return self.f[relidx]
        else:
            return np.nan

    def get_mag_at_frame(self, fr):
        relidx = fr - self.start_idx

        if relidx >= 0:
            return self.mag[relidx]
        else:
            return np.nan

    def synth_no_phase(self, sr, hop, edge=.5):
        '''
        Resynthesise the sinusoidal partial at sampling rate sr
        '''
        # frequency values are delayed by 1/2 frame
        fdel = -hop/float(sr)/2.

        dfr = 1./self.overlap/2.

        # time corresponding to frames
        tfr = (self.start_idx + np.arange(len(self.f) + 2*dfr)) * float(hop)/sr
        ffr = self.f
        ffr = np.insert(ffr, 0, ffr[0]*np.ones(dfr))
        ffr = np.append(ffr, ffr[-1]*np.ones(dfr))
        mfr = self.mag
        mfr = np.insert(mfr, 0, np.linspace(0, mfr[0], dfr+1)[:-1])
        mfr = np.append(mfr, np.linspace(mfr[1], 0, dfr+1)[1:])

        # start synth one frame before and one after to avoid dicontinuites
        tmin = min(tfr) - hop/sr
        tmax = max(tfr) + hop/sr

        # time of samples
        t = np.arange(round(tmin*sr), round(tmax*sr))/sr
        f = np.interp(t + fdel, tfr, ffr)
        mag = np.interp(t, tfr, mfr)

        ph = np.cumsum(2*np.pi*f/sr)

        return mag * np.cos(ph), (self.start_idx)*hop

    def synth(self, sr, hop, intermediate=False, edge=.5):
        nfr = len(self.f)
        # frame delay due to averaging and overlap
        dfr = 1./self.overlap/2.
        sig = np.zeros(hop*(nfr))
        newt = np.arange(hop*(nfr + dfr))
        # try:
        #     oldt = hop*np.arange(nfr + 2*dfr)
        #     mvals = np.append(np.append(self.mag[0]*np.ones(dfr), self.mag), self.mag[-1]*np.ones(dfr))
        #     mfunc = interp1d(oldt - 1.0, mvals, kind = 'cubic')
        #     msig = mfunc(newt)
        #
        #     fvals = np.append(np.append(self.f[0]*np.ones(dfr), self.f), self.f[-1]*np.ones(dfr))
        #     func = interp1d(hop*(-.5 + oldt), fvals,
        #                     kind='cubic')
        #     fsig = func(newt)
        # except NameError:
        fsig = np.interp(newt, hop*(dfr + .5 + np.arange(nfr)), self.f)
        msig = np.interp(newt, hop*(dfr + np.arange(nfr)), self.mag)
        for ii in xrange(nfr):
            # fsam = self.f[ii]*np.ones(hop - 1)
            fsam = fsig[hop*ii:(hop*(ii+1) - 1)]
            # msam = np.interp(self.mag[ii]
            ph = pi2 * np.cumsum(fsam/float(sr))
            ph = np.insert(ph, 0, 0)

            # phase correction for variable frequency
            if self.fstep is None:
                phcor = 0.0
                phcornext = 0.0
            else:
                phcor = np.pi*(fsig[hop*(ii+1)] - fsig[hop*ii])/self.fstep/2.
                if ii < nfr-1:
                    phcornext = np.pi*(fsig[hop*(ii+2)] -
                                       fsig[hop*(ii+1)])/self.fstep/2.

            # starting phase
            ph0 = self.realph[ii] + phcor
            ph += ph0

            if ii < nfr-1:
                # phase correction for discontinuities
                phend = ph[-1] + pi2*fsig[hop*(ii+1)]/float(sr)
                dph = np.mod(self.realph[ii+1] + phcornext - phend+np.pi,
                             pi2) - np.pi
                ph += np.linspace(0.0, dph, num=hop+1)[:-1]

            # import pdb
            # pdb.set_trace()

            thissig = msig[hop*ii:hop*(ii+1)] * np.cos(ph)
            # phsig[hop*ii:hop*(ii+1)] = ph
            sig[hop*ii:hop*(ii+1)] = thissig

        # smoothen edges of partial
        # beginning
        edgsam = int(dfr*hop*edge)
        # mag = np.linspace(0.0, self.mag[0], edgsam)
        mag = msig[0]*(1 - np.cos(np.pi*np.arange(edgsam)/float(edgsam)))/2.
        phb = np.flipud(self.realph[0] -
                        pi2*np.cumsum(self.f[0]*np.ones(edgsam)/float(sr)))
        sig = np.insert(sig, 0, mag*np.cos(phb))
        # end
        # mag = np.linspace( self.mag[-1], 0.0, edgsam)
        mag = msig[hop*(ii+1)]*(1+np.cos(np.pi*np.arange(edgsam)
                                         /float(edgsam)))/2.
        phb = ph[-1] + pi2*np.cumsum(self.f[-1]*np.ones(edgsam)/float(sr))
        sig = np.append(sig, mag*np.cos(phb))

        if intermediate:
            return sig, int((self.start_idx)*hop - edgsam), phsig
        else:
            return sig, int((self.start_idx)*hop - edgsam)

    def get_rel_phase(self):
        nfr = len(self.mag)
        # frame delay due to averaging and overlap
        dfr = 1./self.overlap/2.
        newt = hop*(dfr + np.arange(nfr))
        fsig = np.interp(newt, hop*(dfr + .5 + np.arange(nfr)), self.f)
        thisph = np.zeros_like(self.ph)
        # msig = np.interp(newt, hop*(dfr+np.arange(nfr)), self.mag)
        msig = self.mag
        for ii in xrange(nfr - 1):
            ph = 0.

            # phase correction for variable frequency
            if self.fstep is None:
                phcor = 0.0
                phcornext = 0.0
            else:
                phcor = np.pi*(fsig[(ii+1)] - fsig[ii])/self.fstep/2.
                if ii < nfr-2:
                    phcornext = np.pi*(fsig[(ii+2)] - fsig[(ii+1)])/self.fstep/2.

            # starting phase
            ph0 = self.realph[ii] + phcor
            ph += ph0

            # if ii < nfr-1:
            #     # phase correction for discontinuities
            #     phend = ph[-1] + pi2*fsig[hop*(ii+1)]/float(sr)
            #     dph = np.mod(self.realph[ii+1] + phcornext - phend+np.pi , pi2) - np.pi
            #     ph += np.linspace(0.0, dph, num=hop+1)[:-1]

            # import pdb
            # pdb.set_trace()

            thisph[ii] = ph

        return thisph


class SinSum(object):
    def __init__(self, sr, nfft=1024, hop=512):
        '''
        Sine sum object:
        Represents a sound decomposed in a sum of quasi-sine waves,
        in which amplitude and frequency vary slowly in time
        Arguments:
            * sr   = Sampling rate
            * nfft = Number of points in FFT analysis window
            * hop  = Number of points between FFT windows
        '''

        # sine component structure
        self.partial = []

        # store start and end values for faster search
        self.st = []
        self.end = []
        self.nfft = nfft
        self.hop = hop
        self.sr = sr

    def add_empty_partial(self, idx):
        '''
        Append an empty partial at frame idx
        '''

        newpart = RegPartial(idx, overlap=self.hop/float(self.nfft),
                             fstep=self.sr/float(self.nfft))
        self.partial.append(newpart)
        self.st.append(idx)
        self.end.append(idx)

        return newpart

    def add_point(self, fr, f, mag, ph, maxpitchjmp=0.5):
        '''
        Add a point to the matching partial or create a new one
        Slow! Use add_frame, to add all the peak values
        '''

        # partials = self.get_partials_at_frame(fr-1)
        pidx = self.get_partials_idx_ending_at_frame(fr - 1)
        # tidx = self.get_partials_idx_at_frame(fr)
        # pidx = np.setdiff1d(pidx, tidx, assume_unique=True)

        if len(pidx) > 0:
            pmag = [self.partial[ii].get_mag_at_frame(fr - 1) for ii in pidx]

            # zips, sorts and unzips
            pmag, pidx = zip(*sorted(zip(pmag, pidx), reverse=True))
            partials = [self.partial[ii] for ii in pidx]
            prev_f = [pp.get_freq_at_frame(fr-1) for pp in partials]

            stonediff = np.array([abs(dpitch2st(ff, f)) for ff in prev_f])
            dbdiff = 20*np.log10(np.array(pmag)/mag)
            # partials should be near in frequency and magnitude
            ovdiff = stonediff + abs(dbdiff)
            nearest = np.argmin(ovdiff)

            if stonediff[nearest] < maxpitchjmp:
                idx = pidx[nearest]
                part = partials[nearest]
            else:
                part = self.add_empty_partial(fr)
                idx = -1
        else:
            part = self.add_empty_partial(fr)
            idx = -1

        part.append_point(f, mag, ph)
        self.end[idx] = fr


    def add_frame(self, fr, f, mag, ph, realph=None, maxpitchjmp=0.5):

        # process new peaks in decreasing magnitude
        irev = np.argsort(mag)
        idx = irev[::-1]
        idx = idx[np.logical_and(f[idx] > 0, mag[idx] > 0)]
        fsrt = f[idx]
        msrt = mag[idx]
        psrt = ph[idx]
        if realph is None:
            rsrt = ph[idx]

        else:
            rsrt = realph[idx]

        # partials = self.get_partials_at_frame(fr-1)
        pidx = self.get_partials_idx_ending_at_frame(fr-1)
        # if there are some previous partials...
        if len(pidx) > 0:
            # get all magnitudes in previous frame
            pmag = [self.partial[ii].get_mag_at_frame(fr-1) for ii in pidx]
            # sort partials per magnitude
            allpmagl, pidx = zip(*sorted(zip(pmag, pidx), reverse=True))

            allpidx = np.array(pidx)
            allpmag = np.array(allpmagl)

            allpartials = [self.partial[ii] for ii in allpidx]
            allpf = np.array([pp.get_freq_at_frame(fr-1)
                              for pp in allpartials])
            unused = np.ones_like(allpidx, dtype=bool)

            for fc, mc, pc, rc in zip(fsrt, msrt, psrt, rsrt):
                # select old partials
                pidx = allpidx[unused]
                # print 'unused: %d, len: %d\n'%(sum(unused), len(pidx))

                # if pidx is not empty...
                if len(pidx) > 0:
                    # print 'Checking previous..'
                    pmag = allpmag[unused]
                    pf = allpf[unused]

                    stonediff = abs(dpitch2st(pf, fc))
                    # dbdiff = 20*np.log10(pmag/mc)
                    # very rough formula for db:
                    dbdiff = dbconst*(pmag/mc - 1)

                    ovdiff = stonediff#+abs(dbdiff)
                    nearest = np.argmin(ovdiff)
                    # print 'Distance: %f'%(ovdiff[nearest])

                    if stonediff[nearest] < maxpitchjmp:

                        idx = pidx[nearest]
                        part = self.partial[idx]
                        unuidx = np.nonzero(unused)[0]
                        unused[unuidx[nearest]] = False
                    else:
                        # sys.stderr.write('New partial {} at frame {}: min semitone interval = {}\n'.format
                        #                  (len(self.partial) + 1, fr, stonediff[nearest]))
                        # sys.stderr.write('This frequency: {}\n'.format(fc))
                        # sys.stderr.write('Previous frame frequencies\n')
                        # for ff, uu in zip(allpf, unused):
                        #     sys.stderr.write('{}'.format(ff))
                        #     if uu:
                        #         sys.stderr.write('\n')
                        #     else:
                        #         sys.stderr.write(' (used) \n')

                        part = self.add_empty_partial(fr)
                        idx = -1
                # if no more previous partials left (pidx is empty)
                # add a new partial
                else:
                    part = self.add_empty_partial(fr)
                    idx = -1

                part.append_point(fc, mc, pc, realph=rc)
                self.end[idx] = fr
        else:
            for fc, mc, pc, rc in zip(fsrt, msrt, psrt, rsrt):
                part = self.add_empty_partial(fr)
                idx = -1

                part.append_point(fc, mc, pc, realph=rc)
                self.end[idx] = fr


    def get_partials_at_frame(self, fr):
        '''
        Return the partials at frame fr
        '''
        partials = []

        for idx, ilims in enumerate(zip(self.st, self.end)):
            if fr >= ilims[0] and fr <= ilims[1]:
                partials.append(self.partial[idx])

        return partials

    def get_partials_idx_at_frame(self, fr):
        '''
        Return the partials index at frame fr
        '''
        pidx = []

        for idx, ilims in enumerate(zip(self.st, self.end)):
            if fr >= ilims[0] and fr <= ilims[1]:
                pidx.append(idx)

        return np.array(pidx)

    def get_partials_idx_ending_at_frame(self, fr):
        '''
        Return the index of the partials ending at fr
        '''
        pidx = []

        for idx, ilims in enumerate(zip(self.st, self.end)):
            if fr >= ilims[0] and fr == ilims[1]:
                pidx.append(idx)

        return np.array(pidx)

    def get_points_at_frame(self, fr):
        '''
        Return the parameters of all partials at frame fr
        '''
        pass

    def plot_time_freq(self, minlen=10):
        part = [pp for pp in self.partial if len(pp.f) > minlen]
        pl.figure()
        pl.hold(True)
        for pp in part:
            pl.plot(pp.start_idx + np.arange(len(pp.f)), np.array(pp.f))
        pl.hold(False)
        pl.xlabel('Time (s)')
        pl.ylabel('Frequency (Hz)')
        # pl.show()
        return pl.gca()

    def two_plot_time_freq_mag(self, minlen=10):
        part = [pp for pp in self.partial if len(pp.f) > minlen]
        pl.figure()
        ax1 = pl.subplot(211)
        pl.hold(True)
        ax2 = pl.subplot(212, sharex=ax1)
        pl.hold(True)
        for pp in part:
            ax1.plot(pp.start_idx + np.arange(len(pp.f)), np.array(pp.f))
            ax2.plot(pp.start_idx + np.arange(len(pp.f)),
                     20*np.log10(np.array(pp.mag)))
        ax1.hold(False)
        # ax1.xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        # pl.show()
        return pl.gca()

    def plot_time_freq_mag(self, minlen=10, cm=pl.cm.rainbow):

        cadd = 30
        cmax = 256
        ccur = 0

        part = [pp for pp in self.partial if len(pp.f) > minlen]
        pl.figure()
        pl.hold(True)
        for pp in part:
            # pl.plot(pp.start_idx + np.arange(len(pp.f)), np.array(pp.f))
            mag = 100 + 20*np.log10(np.array(pp.mag))
            pl.scatter(pp.start_idx + np.arange(len(pp.f)), np.array(pp.f),
                       s=mag, c=cm(ccur), lw=0)
            ccur = np.mod(ccur + cadd, cmax)
        pl.hold(False)
        pl.xlabel('Time (s)')
        pl.ylabel('Frequency (Hz)')
        pl.show()

    def synth(self, sr, hop, edge=1.0, minframes=3, phase_preserve=True):
        # edges
        dfr = self.nfft/self.hop/2.
        edgsamp = edge*hop*dfr

        # fixme: why +2???
        w = np.zeros((max(self.end) + 2)*hop + 2*edgsamp)
        for part in self.partial:
            if len(part.f) >= minframes:
                if phase_preserve:
                    wi, spl_st = part.synth(sr, hop, edge=edge)
                else:
                    wi, spl_st = part.synth_no_phase(sr, hop, edge=edge)
                spl_st += edgsamp
                if spl_st >= 0:
                    spl_end = spl_st + len(wi)
                    w[spl_st:spl_end] += wi
        return w[edgsamp:]

    def get_avfreq(self):
        return np.array([np.mean(xx.f) for xx in self.partial])

    def get_avmag(self):
        return np.array([np.mean(xx.mag) for xx in self.partial])

    def get_summary(self, minlen=10):
        psum = np.array([(ii, len(xx.f), np.mean(xx.f),
                          np.mean(xx.mag)) for ii,
                         xx in enumerate(self.partial) if len(xx.f) > minlen],
                        dtype=[('idx', 'i4'), ('n', 'i4'),
                               ('f', 'f4'), ('mag', 'f4')])
        psum.sort(order='mag')
        return psum

    def get_nframes(self):
        # return max([xx.start_idx + len(xx.mag)])
        return max(self.end)

    def get_part_data_around_freq(self, fc, semitones=.5):
        nframes = self.get_nframes() + 1
        t = np.arange(nframes)/float(self.sr)*self.hop
        f = np.zeros(nframes)
        mag = np.zeros(nframes)
        ph = np.zeros(nframes)

        ssa = self.get_summary(minlen=0)
        ssa.sort(order='mag')
        # ssd = np.flipud(ssa)
        ss = ssa
        idx = ss['idx'][(abs(12*np.log2(abs(ss['f']/fc))) < semitones).nonzero()]

        for i in idx:
            sti = self.partial[i].start_idx
            endi = self.partial[i].start_idx + len(self.partial[i].mag)
            f[sti:endi] = self.partial[i].f
            mag[sti:endi] = self.partial[i].mag
            ph[sti:endi] = self.partial[i].ph

        return t, f, mag, ph
