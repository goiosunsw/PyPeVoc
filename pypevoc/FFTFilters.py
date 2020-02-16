#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Filters.py
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

import numpy as np

class BandError(Exception):
    """Exception raised for errors in band definition.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self,  message):
        self.message = message
        Exception.__init__(self, message)


def preemph(w,hpFreq=0,Fs=1):
    '''
    Applies a pre-emphasis filter to the signal w
        amplifies the signal with a +6dB/octave
        filter above the cut-on frequency
        
    Arugments:
    * hpFreq = cut-on frequency
    * Fs = sampling frequency
    '''
    
    if hpFreq>0:
        a=np.exp(-2.*np.pi*hpFreq/float(Fs));
        #preEmphA = [a,1-a];
        #wo = sig.lfilter([1],preEmphA,w);
        wo=w.astype('f')
        wo[:-1] -= wo[1:]*a
    else:
        wo=w
    return wo

def _f_to_mel_py(freq):
    # mel = 1125 * ln(1+f/700)
    return 1125. + np.log(1.+freq/700.)

def _mel_to_f_py(mel):
    return 700.*(np.exp(mel-1125.)-1)

f_to_mel = np.vectorize(_f_to_mel_py)
mel_to_f = np.vectorize(_mel_to_f_py)


def peaks(x):
    '''
    Return indexes of all local maxima in x
    '''
    pkmask = np.logical_and(x[:-2]<x[1:-1],x[2:]<x[1:-1])
    return np.flatnonzero(pkmask)+1

    
def nearest(a, b):
    '''Find elements of b nearest to elements of a'''
    an = np.zeros(len(a))
    for ia,aa in enumerate(a):
        idx = np.argmin(np.abs(b-aa))
        an[ia]=b[idx]
    return an

class PiecewiseFilterSpec(object):
    '''
    Builds and stores specifications for a FFT filter
    '''
    bandf = np.array([0.0, 0.5])
    bandg = np.array([1.0, 1.0])
    sr = 1.0
    label=''
    
    def __init__(self, mode='', cutoff=0.5, 
                 freq = np.array([0.0, 0.5]),
                 gain = np.array([1.0, 1.0]),
                 sr = 1.0,
                 label = ''):
        '''
        Create a new filter specification
        
        Can be called using the following presets passed as mode:
        * Lowpass or lp: value in freq argument is cutoff
        * Hipass or hp: value in freq argument is cuton
        * Bandpass or bp: list in freq argument is cutoff and cuton
        * Bandstop or bs: list in freq argument is cuton and cutoff
        
        Otherwise provide frequency vertexes and corresponding gains
        in freq and gain arguments
        
        Frequencies are divided by sr
        
        Optionnaly provide a label for the filter
        '''
        
        self.sr=sr
        
        if mode.lower()=='lp' or mode.lower()=='lowpass':
            self.set_lowpass_cutoff(freq/float(sr))
        elif mode.lower()=='hp' or mode.lower()=='hipass' or mode.lower()=='highpass':
            self.set_hipass_cutoff(freq/float(sr))
        elif mode.lower()=='bp' or mode.lower()=='bandpass':
            self.set_bandpass_freqs(freq[0]/float(sr), freq[-1]/float(sr))
        elif mode.lower()=='bs' or mode.lower()=='bandstop':
            self.set_bandstop_freqs(freq[0]/float(sr), freq[-1]/float(sr))
        else:
            assert len(freq) == len(gain)
            self.set_triangular_filter(freq,gain)
            self.label = label
        
        if not self.label:
            self.label = 'Piecewise filter with {} bands'.format(len(self.bandf)-1)

    def set_lowpass_cutoff(self,f):
        self.bandf = np.array([[0.0, f],[f, 0.5]])
        self.bandg = np.array([[1.0, 1.0],[0.0, 0.0]])
        self.label = 'Lowpass filter, fc={}'.format(f*self.sr)
        
    def set_hipass_cutoff(self,f):
        self.bandf = np.array([[0.0, f],[f, 0.5]])
        self.bandg = np.array([[0.0, 0.0],[1.0, 1.0]])
        self.label = 'Hipass filter, fc={}'.format(f*self.sr)

    def set_bandpass_freqs(self,f1,f2):
        self.bandf = np.array([[0.0, f1],[f1, f2],[f2,0.5]])
        self.bandg = np.array([[0.0, 0.0],[1.0, 1.0],[0.0, 0.0]])
        self.label = 'Bandpass filter, fc={}'.format((f1/2+f2/2)*self.sr)

    def set_bandstop_freqs(self,f1,f2):
        self.bandf = np.array([[0.0, f1],[f1, f2],[f2,0.5]])
        self.bandg = np.array([[1.0, 1.0],[0.0, 0.0],[1.0, 1.0]])
        self.label = 'Bandstop filter, fc={}'.format((f1/2+f2/2)*self.sr)
    
    def set_triangular_filter(self,freq,gain):
        bands = []
        bgains = []
        idx = np.argsort(freq)
        for iprev,inext in zip(idx[:-1],idx[1:]):
            bands.append([freq[iprev]/self.sr,freq[inext]/self.sr])
            bgains.append([gain[iprev],gain[inext]])
            
        self.bandf = np.array(bands)
        self.bandg = np.array(bgains)
        

        
    def __repr__(self):
        rep =  '{}:\n'.format(self.label)
        for f,g in zip(self.bandf,self.bandg):
            fst = f[0]*self.sr
            fend = f[1]*self.sr
            if g[0] == g[1]:
                rep+='  Freq = [{},{}]: gain = {}\n'.format(fst,fend,g[0])
            else:
                rep+='  Freq = [{},{}]: gain = [{},{}]\n'.format(fst,fend,g[0],g[1])
                
        return rep
        
    def get_frequency_gains(self):
        '''
        Returns frequency values and gains correspoinding to the
        piecewise filter.
        
        fband, bandg = get_frequency_gains(self)
        
        fband and bandg ar Nx2 arrays
        '''
        return np.array(self.bandf)*self.sr, np.array(self.bandg)

    def get_frequency_edges(self):
        '''
        Returns the unique values of frequency edges
        '''
        return np.unique((np.array(self.bandf).flatten()*self.sr))

    
    def apply_to_freq_vector(self,fvec, align_edges=False):
        '''
        Returns the values of gain at frequencies in fvec
        '''
        fvec=np.array(fvec)
        flim = self.get_frequency_edges()
        edge_dict=dict()
        if align_edges:
            for ff in flim:
                idx = np.argmin(np.abs(fvec-ff))
                edge_dict[ff] = fvec[idx]
        else:
            for ff in flim:
                edge_dict[ff] = ff
        
        #print edge_dict
        
        filter_mask = np.zeros(len(fvec))
        freqs = self.bandf*self.sr
        for f,g in zip(freqs,self.bandg):
            fst = edge_dict[f[0]]
            fend = edge_dict[f[1]]
            idx = np.logical_and(fvec>=fst,
                                 fvec<=fend)
            if fend!=fst:
                filter_mask[idx]=(fvec[idx]-fst)/(fend-fst)*(g[1]-g[0])+g[0]
            else:
                raise BandError('Band is too narrow: try increasing nwind')
                    

        return filter_mask

        
        

class FilterBank(object):
    '''
    FilterBank object: Defines a FFT-based filter bank
    '''
    label = []
    fvec = np.zeros(0)
    fb=np.zeros((0,0))
    sr=1.
    
    def __init__(self, fspec_list=None, sr=1.0, 
                 nwind=256, windfunc=np.hanning,
                 nhop=None, align_edges=True):
        '''
        Create a filter bank from a list of filter specification 
          objects PiecewiseFilterSpec
          
        By default creates a 2-band filterbank 
          dividing the range [0,sr/2] into two bands 
        '''
        self.sr = sr
        self.wind  = windfunc(nwind)
        self.nwind = int(nwind)
        if nhop:
            self.hop = nhop
        else:
            self.hop = int(nwind/2)

        self.fvec = np.linspace(0.,sr,nwind)
        if not fspec_list:
            fc=0.25
            fspec_list=[PiecewiseFilterSpec(mode='lowpass',freq=fc,sr=sr),
                        PiecewiseFilterSpec(mode='hipass',freq=fc,sr=sr)]
                        
        self.fb = np.zeros((len(fspec_list),len(self.fvec)))
        self.label=[]
        for ii,fspec in enumerate(fspec_list):
            self.fb[ii,:]=fspec.apply_to_freq_vector(self.fvec,align_edges=align_edges)
            self.label.append(fspec.label)
            
    def specout(self,w):
        '''
        Calculate the output of the filterbank applied to w
        '''

        n=0
        bankout = []
        tout=[]
        while n<len(w)-self.nwind:
            thisbank = []
            ww = w[n:n+self.nwind]*self.wind
            Sw = np.fft.fft(ww)
            Sww = np.abs(Sw)**2
            for i in range(self.fb.shape[0]):
                thisbank.append(sum(Sww*self.fb[i,:]))
            bankout.append(thisbank)
            tout.append((float(n)+self.nwind/2.)/float(self.sr))
            n+=self.hop
        return np.array(bankout),np.array(tout)

    def __repr__(self):
        rstr = 'FilterBank with filters:\n'
        for ll in self.label:
            rstr+='  '+ll+'\n'
        return rstr

class TriangularFilterBank(FilterBank):
    '''
    FilterBank object: Defines a FFT-based filter bank
    '''
    label = []
    fvec = np.zeros(0)
    fb=np.zeros((0,0))
    sr=1.

    def __init__(self,flim=[0,.5,1.],nwind=256, sr=1., nhop=None):
        '''
        Create a filter bank:
        * flim:  limits of frequency bands
        * nwind: window for FFT
        * nhop:  interval between successive filtered frames 
                 (half window by default)
        * sr:    sampling rate (by default = 1, in that case define flim
            as a fraction of sampling rate [0,1)
        '''
        
        fsl=[]
        
        if sr>1.0:
            unit = 'Hz'
        else:
            unit = ''
        
        flim = np.sort(flim).astype('f')
        for n,cc in enumerate(flim[1:-1]):
            bandf = flim[n:n+3]
            bandg = np.array([0.0,1.0,0.0])
            lab = '{}{} band ({}-{}{})'.format(cc,unit,flim[n],flim[n+2],unit)
            fsl.append(PiecewiseFilterSpec(freq=bandf,gain=bandg,label=lab,sr=sr))
             
        super(TriangularFilterBank,self).__init__(fspec_list=fsl,nwind=nwind,sr=sr,nhop=nhop)


def nextpow2(x):
    return 2**(np.ceil(np.log2(x)))



class MelFilterBank(TriangularFilterBank):
    def __init__(self,n=26,fmin=300.,fmax=8000.,twind=.025, sr=44100., thop=.01):
        nwind = int(2**np.round(np.log2(twind*sr)))
        nhop = int(thop*sr)
        melmin = f_to_mel(fmin)
        melmax = f_to_mel(fmax)
        fc = mel_to_f(np.linspace(melmin,melmax,n+2))

        super(MelFilterBank,self).__init__(flim=fc,nwind=nwind,sr=sr,nhop=nhop)

    def mfcc(self,w,mode='DCT2'):
        spec, tspec = self.specout(w)
        logs = np.log(spec)
        if mode[:3]=='DCT':
            dctype = int(mode[3])
            from scipy.fftpack import dct
            return dct(logs,type=dctype), tspec
        elif mode=='IFFT':
            return np.fft.ifft(logs), tspec
        else:
            raise NotImplementedError
    
def fft_filter(x, bands, gains):
    '''
    Filter signal x using FFT and IFFT
    * x input signal
    * bands: list of start and stop frequencies of each band
    * gains: start and stop gains in each band
    
    Example:
    
    y = FFTfilter(x, [(0,0.1),(0.1,1.0)], [(1.,1.),(0.,0.)])
    
    filters signal x low pass at 0.1 times the nyquist rate
      (sampling rate / 2)
    '''
    
    xf = np.fft.fft(x)
    nyq = len(xf)/2
    
    ffilter = np.zeros(len(xf))
    for bb, gg in zip(bands,gains):
        fmin = int(bb[0]*nyq)
        fmax = int(bb[1]*nyq)
        ffilter[fmin:fmax]=np.linspace(gg[0],gg[1],fmax-fmin)
        if fmin>0:
            ffilter[-fmax+1:-fmin+1]=np.linspace(gg[1],gg[0],fmax-fmin)
        else:
            ffilter[-fmax+1:]=np.linspace(gg[1],gg[0],fmax-fmin-1)
        print('{}-{} : gains [{}, {}]'.format(fmin,fmax,gg[0],gg[1]))
        
    xf_filt = xf*ffilter
    return np.fft.ifft(xf_filt) 

