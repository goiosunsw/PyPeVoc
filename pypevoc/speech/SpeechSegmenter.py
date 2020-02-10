#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  SpeechSegmenter.py
#  
#  Copyright 2017 Andre Almeida <goios@goios-UX305UA>
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
from . import SpeechAnalysis as sa
from .. import FFTFilters as ff
from .. import PeakFinder as pkf

def peaks(x):
    '''
    Return indexes of all local maxima in x
    '''
    pkmask = np.logical_and(x[:-2]<x[1:-1],x[2:]<x[1:-1])
    return np.flatnonzero(pkmask)+1

def refine_transition_points(x, tx=None, trt=None, mode='minmax', thr=0.5):
    '''
    Finds the transition point closer to trt as the crossing 
    of the threshold between typical values before and after 
    the rough transition trt.
    
    Typical values can be minimum and maximum or percentile values
    
    Arguments:
    x:   time series
    tx:  time values corresponding to x series
    trt: rough transition time
    thr: level betwen typical value before and after
         e.g. 0.5 for halfway between lower and higher values
    mode: 
      * minmax: minimum and maximum before and after trt
      * median: medians before and after
      * mean: means before and after
      * pctYY: percentiles YY and 1-YY before and after
    '''
    
    if tx is not None:
        # Make sure time series is sorted
        istx = np.argsort(tx)
        tx = np.array(tx)[istx]
        x  = np.array(x)[istx]
    else:
        x  = np.array(x)
        tx = np.arange(len(x))
    
    if trt>max(tx) or trt<min(tx):
        return trt,0,0
    
    
    xbef = x[tx<trt]
    xaft = x[tx>trt]
    # index of rough transition pont
    irtr = np.flatnonzero(tx>trt)[0]
    
    if mode == 'minmax':
        if np.median(xbef)>np.median(xaft):
            vbef = np.max(xbef)
            vaft = np.min(xaft)
        else:
            vbef = np.min(xbef)
            vaft = np.max(xaft)
    elif mode[:3] == 'pct':
        pct = np.float(mode[3:])
        if np.median(xbef)>np.median(xaft):
            vbef = np.percentile(xbef,100-pct)
            vaft = np.percentile(xaft,pct)
        else:
            vbef = np.percentile(xbef,pct)
            vaft = np.percentile(xaft,100-pct)
    elif mode == 'median':
        vbef = np.median(xbef)
        vaft = np.median(xaft)
    elif mode == 'mean':
        vbef = np.mean(xbef)
        vaft = np.mean(xaft)
        
    if vaft>vbef:
        # transition value
        vtran = vbef*thr+vaft*(1-thr)
        
        # find crossings (change in sign of x-vtran)
        itr_all = np.flatnonzero(np.logical_and(x[:-1]<vtran,x[1:]>vtran))
    else:
        # transition value
        vtran = vaft*thr+vbef*(1-thr)
        
        # find crossings (change in sign of x-vtran)
        itr_all = np.flatnonzero(np.logical_and(x[:-1]>vtran,x[1:]<vtran))
        
        
    # time values corresponding to transitions
    ttr_all = (tx[itr_all]+tx[itr_all+1])/2.
    
    # keep the nearest to  rough transition
    ii = np.argmin(np.abs(ttr_all - trt))
    itr = itr_all[ii]
    
    # refine transition time (interpolation)
    ttr = tx[itr] + (tx[itr+1]-tx[itr])*(vtran-x[itr])/(x[itr+1]-x[itr])
     
    return ttr, vbef, vaft


class SpeechSegmenter(object):
    '''
    Speech segmenter object 
    '''
    
    def __init__(self,sr=44100., bands = [225.,2000.,4000.,8000.,15000.],
                      refine_bands = None,
                      detect_thresh = 0.7,
                      rough_window = 2048, fine_window = 256):
        '''
        Create the segmeter object:
        sr:    sampling rate
        bands: filterbank bands for estimation of rough segments
        rough_window: rough window for the segmentation function
        refine_bands: filterbanks for  refinement of segments
        fine_window:  window for refinement of segmentation
        detect_thresh: threshold for rate of change in detection function
               (db/sec)
        '''
        
        rough_lims = bands   
        if sr/2. not in rough_lims:
            rough_lims.append(sr/2.)
            rough_lims = np.array(rough_lims)
            
        self.sr = sr
        
        self.rough_window =rough_window
        self.fine_window  = fine_window
        
        self.pkthresh = self.sr/self.rough_window*detect_thresh
        
        self.bands=bands
        self.rough_bands = bands
        self.refine_bands = refine_bands
    
    def set_signal(self, signal, sr=1):
        '''
        set a new sound signal to analyse
        '''
        self.sig = signal
        self.sr = sr
        self.set_bands(self.bands)
        self.tmax = len(signal)/float(sr)

    def set_soundfile(self, filename):
        oldsr = self.sr
        try:
            from scipy.io import wavfile
            
            sr,w = wavfile.read(filename)
            self.set_signal(w, sr)
        except ImportError:
            raise
            
        if self.sr != oldsr:
            self.set_bands(self.bands)
        self.tmax = len(w)/float(sr)
            
    def set_bands(self, bands):
        
        self.bands = bands
        self.rough_fb = ff.TriangularFilterBank(flim=bands,
                                                sr=self.sr,
                                                nwind=self.rough_window)
    
        if not self.refine_bands:
            self.refine_bands = self.bands
            
        
        self.fine_fb = ff.TriangularFilterBank(flim=self.refine_bands,
                                               sr=self.sr,
                                               nwind=self.fine_window)

 
    
    def process(self, w):
        '''
        Process the signal w, finding rough segmentation points
        '''
        self.sig = w
        
        fbspec,tfb = self.rough_fb.specout(w)
        dbspec = 10*np.log10(fbspec)
        ddbspec = np.sum(np.abs(np.diff(dbspec,axis=0)),axis=1)
        tdfb = tfb[:-1] + self.rough_fb.hop/2/float(self.sr)
        
        ddbpk = peaks(ddbspec)
        ddbpk = ddbpk[ddbspec[ddbpk]>self.pkthresh]
        
        self.detection_func = ddbspec
        self.detection_func_times = tdfb
        self.detection_func_indexes_int = ddbpk
        self.segments = tfb[ddbpk]
        
        return self.segments
    
    def refine_segment_parabolic_fit(self, pos):
        '''
        Refines the segment boundaries 
        based on a polyfit on the detection function
        
        pos: index of segment in the detection funtion
        '''
        
        
        sur = self.detection_func[pos-1:pos+2]
        
        
        if sur[1]>sur[0] and sur[1]>sur[2]:
            c = sur[1]
            b = (sur[2] - sur[0])/2
            a = (sur[2] + sur[0])/2 - c
        
            lpos = - b/2/a 
            fpos = float(pos) + lpos 
            fval = a*lpos*lpos + b*lpos + c
            #print "rpos = %d; rval = %f; val = %f; dpos = %f; pos = %f"%(pos,sur[1],fval, lpos, fpos)
                
        else:
            fpos = pos
            fval = sur[1]
            
        return fpos,fval.tolist()

    def refine_segments_parabolic(self):
        self.detection_function_indexes_ref = []
        for iseg,pk in enumerate(self.detection_func_indexes_int):
            pkref,pkval = self.refine_segment_parabolic_fit(pk)
            self.detection_function_indexes_ref.append(pkref)
            
        self.segments = np.interp(self.detection_function_indexes_ref, 
                                  np.arange(len(self.detection_func_times)),
                                  self.detection_func_times)
    
    def refine_segment_all_bands(self, tseg, thr=None):
        '''
        Refine one peak based on the fine band analysis
        '''
        
        # interval around peak in which to estimate transition values
        intbef = (self.rough_window*2)/float(self.sr)
        
        finespec,tfine = self.fine_fb.specout(self.sig)
        dbfine = 10*np.log10(finespec)
        self.fine_spec = dbfine
        self.fine_time = tfine
        # find transition points in each band
        
        ibef = np.flatnonzero(np.logical_and(tfine>tseg-intbef,tfine<tseg))
        iaft = np.flatnonzero(np.logical_and(tfine>tseg,tfine<tseg+intbef))
        iall = np.flatnonzero(np.logical_and(tfine>tseg-intbef,tfine<tseg+intbef))
        #print('Rough peak at: {}'.format(tseg))
        bandpk = []
        bandweight = []

        for ibd in range(dbfine.shape[1]):
            valb = np.median(dbfine[ibef,ibd])
            vala = np.median(dbfine[iaft,ibd])
            if not thr:
                valm = (valb+vala)/2.
            else:
                valm = np.min((vala,valb))+thr
            thisfine = dbfine[iall,ibd]
            rr = np.flatnonzero((thisfine[:-1]-valm)*(thisfine[1:]-valm)<0)
            if len(rr)>0:
                ir = np.argmin(np.abs(rr-len(ibef+1)))
                pkfine = tfine[rr[ir]+min(ibef)]*self.sr
            else:
                pkfine = tseg
            bandpk.append(pkfine)
            bandweight.append(np.abs(valb-vala))
            #print('Band {} transition at: {} (from {} to {})'.format(
            #    ibd,pkfine,valb,vala))
            #ax[1].axvline(pkfine,ls='-',color=colors[ibd],alpha=.7)
        return np.sum(np.array(bandpk)*np.array(bandweight))/np.sum(bandweight)

    def refine_all_all_bands(self, thr=None):
        '''
        Refine all segments based on the fine band analysis
        '''
        
        newseg=[]
        for seg in self.segments:
            newseg.append(self.refine_segment_all_bands(seg,thr=thr)/self.sr)
        
        self.segments = newseg
        return newseg
        
    def refine_interval(self, tstart, tend, 
                        marg=0.1, thr = 0.5,
                        mode = 'minmax'):
        '''
        Given a rough interval [tstart:tend], refine boundaries 
        of stable sound based on band energies
        
        * tstart, tend: rough interval in source
        * marg: margin around interval to consider for boundary search
        * thr: threshold between values before and after to consider
        * mode: mode of extraction of previous and foloowing values
          - minmax (default) 
          - median
          - pctXX (percentile XX and 100-XX)
        '''
        tmin = tstart-marg
        tmax = tend+marg
        dur = tend-tstart
        
        imin = max(0,int(tmin*self.sr))
        imax = min(len(self.sig),int(tmax*self.sr))
        
        
        x= self.sig[imin:imax]
        finespec,tfine=self.fine_fb.specout(x)
        #tfine += stsec-marg
        dbfinespec = 10*np.log10(finespec*(self.rough_window/self.fine_window)**2)
        # select only extreme bands
        #dbfinespec = dbfinespec[:,[0,-2,-1]]


        ist = np.flatnonzero(np.logical_and(tfine>=0,
                                            tfine<=marg+dur))
        iend = np.flatnonzero(np.logical_and(tfine>=marg,
                                             tfine<=2*marg+dur))
        bandst = []
        weightst = []
        bandend = []
        weightend = []
        
        
        for ibd in range(dbfinespec.shape[1]):
            tst,dbsbef, dbsaft = refine_transition_points(x=dbfinespec[ist,ibd], 
                                   tx = tfine[ist], trt=marg,
                                   mode=mode, thr=thr)
            bandst.append(tst)
            weightst.append(np.abs(dbsaft-dbsbef))
            tend,dbebef,dbeaft = refine_transition_points(x=dbfinespec[iend,ibd], 
                                   tx = tfine[iend], trt=dur+marg,
                                   mode=mode, thr=thr)
            bandend.append(tend)
            weightend.append(np.abs(dbeaft-dbebef))
            
            
            #~ sys.stderr.write(
                #~ 'Band {} transition at: {} (from {} to {})\n'.format(
                    #~ ibd,tst,dbsbef,dbsaft))
            #~ sys.stderr.write(
                #~ 'Band {} transition at: {} (from {} to {})\n'.format(
                #~ ibd,tend,dbebef,dbeaft))
                
        tst = np.sum(np.array(bandst)*np.array(weightst))/np.sum(weightst)
        tend = np.sum(np.array(bandend)*np.array(weightend))/np.sum(weightend)
            
                              
        #tfinal= np.concatenate((tfinal,np.array([[tst,tend],]) + stsec -marg),axis=0)
        tfinal = np.array([tst,tend]) + tstart -marg
        
        self.tfine    = tfine+tmin
        self.finespec = finespec
        self.bandtrans= np.vstack([bandst,bandend])
        self.trans    = tfinal
        
        return tfinal
    
    def plot_last(self):
        import matplotlib.pyplot as pl
        from matplotlib.mlab import specgram
        
        nwind = 256
        novl = nwind/2
        
        fig, ax = pl.subplots(3, sharex=True)
        
        tmin = min(self.tfine)
        tmax = max(self.tfine)
        
        idx = np.arange(int(tmin*self.sr),
                        int(tmax*self.sr))
        thisw = self.sig[idx]
        ax[0].plot(idx/float(self.sr)-tmin,self.sig[idx])
        ax[2].specgram(thisw,Fs=self.sr,NFFT=nwind,noverlap = novl)
        for n in range(self.finespec.shape[1]):
            ln=ax[1].plot(self.tfine-tmin, 20*np.log10(self.finespec[:,n]))
            for axi in ax:
                for i in range(self.bandtrans.shape[0]):
                    axi.axvline(self.bandtrans[i,n],
                                color=ln[0].get_color())
        for axi in ax:
            for tt in self.trans:
                axi.axvline(tt-tmin, color='black')
            
class SyllableSegmenter(object):
    '''
    Segments voice file by amplitude variations
    '''
    def __init__(self,w, sr=1, salience_db=2, voice_intervals=None):
        '''
        Create segmenter object with a wavefile
        '''
        self.w  = w
        self.sr = sr
        if not voice_intervals:
            self.segment_silences()
        else:
            self.intervals = voice_intervals
        
    def segment_amplitude_bumps(self, salience_db=2, 
                                window_sec=0.128, hop_sec=0.016):
                                    
        nwind = int(window_sec*self.sr)
        nhop  = int(hop_sec*self.sr)
        sys.stderr.write('Calculating envelope...\n')

        ta,wa = sa.rmsWind(self.w,nwind=nwind,nhop=nhop,
                           sr=self.sr,windfunc=np.hanning)
        da = 20*np.log10(wa)
        sys.stderr.write('Finding peaks...\n')

        pks=pkf.PeakFinder(da)
        pks.filter_by_salience_from_reach(salience_db)
        pka=pks.get_pos()
        
        
        pki   =  []
        pkpos =  []
        vlst  =  []
        vlend =  []
        pksil =  []
        pkdist = []
        disttype=[]
        utidx =  []
        
        intarray = np.array(self.intervals)
        
        prevint = -1
        
        tpklst = ta[pka]
        
        for iint,(ist,iend) in enumerate(self.intervals):
            pkin = pka[np.logical_and(tpklst>ist, tpklst<iend)]
            #vlst.append(ist)
            tprev = ist
            intervalmask = np.logical_and(ta>ist, ta<iend)
            if sum(intervalmask):
                pkprev=np.flatnonzero(intervalmask)[0]
            else:
                pkprev=0
            
            #print pkprev
            sys.stderr.write('Processing interval {:.3f}-{:.3f}\n'.format(
                              ist,iend))
            pk = pka[0]
            for ii, pk in enumerate(pkin):
                tpk = ta[pk]
                pkdist.append(ta[pk] - tprev)
                pki.append(pk)
                pkpos.append(ta[pk])
                utidx.append(iint)
                vl,vr = pks.get_intervening_minimum(pk)
                dt=0
                if ii==0:
                    vlst.append(ist)
                    dt=-1
                else:
                    vlst.append(ta[vl])
                if ii==len(pkin)-1:
                    vlend.append(iend)
                    dt=1
                else:
                    vlend.append(ta[vr])
                
                
                disttype.append(dt)
                pkprev = pk
                tprev = ta[pk]
                
            
            pkdist.append(iend-ta[pk])
            
        
            
        self.peak_indices = np.array(pki)
        self.peak_positions = np.array(pkpos)
        self.sylable_start = np.array(vlst)
        self.sylable_end = np.array(vlend)
        
        self.peak_distances = np.array(pkdist)
        self.distance_types = np.array(disttype)
        self.parent_interval = np.array(utidx)
        
    def classify_voicing(self, threshold=0.45, fmin=75, fmax=600):
        import Periodicity as per
        pp=per.PeriodTimeSeries(self.w,sr=self.sr,vthresh=threshold,
                                fmax=fmax,fmin=fmin,
                                window=np.hanning(int(self.sr/fmin*3)),
                                hop=(int(self.sr/fmin)),
                                cand_method='similar',
                                threshold=threshold)
        pp.calc()
        period_times = pp.get_times()
        period_voicing = pp.get_strength()
        period_f0 = pp.get_f0()
        self.voicing = period_voicing[np.searchsorted(period_times,
                                                      self.peak_positions)]
        voiced_fraction = []
        mean_f0 = []
        mean_df0 = []
        
        dtpitch = np.min(np.diff(period_times))
        
        for ii,(ts,te) in enumerate(zip(self.sylable_start,self.sylable_end)):
            idx = np.logical_and(period_times >= ts,
                                 period_times <= te)
            mean_f0.append(np.nanmean(period_f0[idx]))
            mean_df0.append(np.nanmean(np.diff(period_f0[idx])))
            voiced_fraction.append(np.mean(period_voicing[idx]>threshold))
            
        self.mean_f0=np.array(mean_f0)
        self.mean_df0=np.array(mean_df0)
        self.voiced_fraction=np.array(voiced_fraction)
            
        self.vthresh = threshold
        
    
    def to_textgrid(self,filename):
        # voicing threshold
        try:
            voicing = self.voicing
            vthresh = self.vthresh
        except AttributeError:
            voicing = np.ones(self.period_positions)
            vthresh = 0.01
        
        from pympi import TextGrid
        tg=TextGrid(xmax=len(self.w)/float(self.sr))
        speech_tier=tg.add_tier('speech')
        for ii, (ts,te) in enumerate(self.intervals):
            speech_tier.add_interval(ts,te,'{}'.format(ii))
        tierp=tg.add_tier('peak',tier_type='TextTier')
        tiers=tg.add_tier('syllable',tier_type='IntervalTier')
        tiers=tg.add_tier('voicing',tier_type='IntervalTier')
        for no,(pk,st,end,vv) in enumerate(zip(self.peak_positions,
                                            self.sylable_start,
                                            self.sylable_end,
                                            voicing)):
            if vv >= vthresh:
                tierp.add_point(pk,'{}'.format(no))
            tiers.add_interval(st,end,'{}'.format(no))
        tg.to_file(filename)

        
        
    def segment_silences(self,window_sec=0.016, hop_sec=0.002,
                         method = 'kmeans',
                         min_len=0.3, max_len=10):
        '''
        Calculates silence and voiced intervals 
        
        Arguments:
        * method = 'kmeans' or 'pctXX' 
        * min_len= minimum length of silence in seconds
        * max_len= maximum length of voiced part in seconds
        '''
        nwind = int(window_sec*self.sr)
        nhop  = int(hop_sec*self.sr)
        
        import SpeechChunker as sc
        co = sc.SilenceDetector(self.w, sr=self.sr, method = method,
                                min_len=min_len, max_len=max_len)

        self.intervals = np.array(
                           [(st,end) for st,end in zip(co.tst, co.tend)])
    
    def to_pandas(self):
        try:
            voicing = self.voicing
            vthresh = self.vthresh
        except AttributeError:
            voicing = np.ones(self.period_positions)
            vthresh = 0.01

        import pandas as pd
        df = pd.DataFrame({'tmax':self.peak_positions,
                           'tstart': self.sylable_start,
                           'tend': self.sylable_end,
                           'voicing': voicing,
                           'utterance': self.parent_interval,
                           'type': self.distance_types,
                           'mean_f0': self.mean_f0,
                           'mean_df0dt': self.mean_df0,
                           'voice_fract': self.voiced_fraction
                           })
        return df
    
def main(args):
    from scipy.io import wavfile 
    
    sr,w16 = wavfile.read(args.infile)
    w=w16.astype('f')/np.iinfo(w16.dtype).max
    
    
    # down to single channel
    if len(w.shape)>1:
        nchan = w.shape[1]
        w=np.mean(w,axis=1)

    
    if args.start>0.0:
        ist = int(args.start*sr)
    else:
        ist = 0
        
    if args.end>0.0:
        iend = int(args.end*sr)
    else:
        iend = len(w)
    
    w=w[ist:iend]
    
    bands = [225,2000,4000,8000,15000,sr/2]
    
    sseg = SpeechSegmenter(sr=sr, bands = bands)
    tseg = sseg.process(w)
    tsegr = sseg.refine_all_all_bands()
    tsegr = np.array(tsegr) + float(ist)/sr
    
    if args.stdout:
        for tt in tsegr:
            print(tt)
    
    return 0

if __name__ == '__main__':
    import sys
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", nargs='?', default = '',
        help = "output file name")
    ap.add_argument("-b", "--bands", type=int, nargs='?', default = [],
        help = "center frequency of bands")

    ap.add_argument("-s", "--start", type=float, nargs='?', default = '0',
        help = "start time")
    ap.add_argument("-e", "--end", type=float, nargs='?', default = '-1',
        help = "end time")


    ap.add_argument("-c", "--csv", action='store_true',
        help = "output a CSV file")
    ap.add_argument("-t", "--stdout", action='store_true',
        help = "output to screen")
    ap.set_defaults(csv=False)
    ap.set_defaults(stdout=True)
    
    ap.add_argument('infile', nargs='?', help='Input sound file (required)')

    args = ap.parse_args()

    sound_file = args.infile

    

    sys.exit(main(args))


