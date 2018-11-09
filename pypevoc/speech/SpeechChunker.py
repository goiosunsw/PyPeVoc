#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  SpeechChunkerer.py
#  
#  Copyright 2017 Andre Almeida <a.almeida@unsw.edu.au>
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
import os

import numpy as np

from .SpeechAnalysis import rmsWind

try:
    from scipy.io.wavfile import read as wavread
    from scipy.io.wavfile import write as wavwrite
except ImportError:
    sys.stderr.write('Scipy wav reader not found!\nUsing internal reader\n')
    from AudioInterface import wavLoad as wavread
    from AudioInterface import wavWrite as wavwrite

class SilenceDetector(object):
    '''
    Detects regions of silence in a sound file
    '''
    
    def __init__(self, x, sr=1, wind_sec=0.92, method = 'pct05',
                 min_len = 0.1, max_len=5, fmin=None, fmax=None):
        '''
        crate silence detector
        '''
        self.x=x
        self.sr=sr
        self.nwind=int(wind_sec*sr)
        
        if fmin is None and fmax is None:
            self._calc_amplitude(nwind=self.nwind)
        else:
            self._calc_band_amplitude(nwind=self.nwind, fmin=fmin, fmax=fmax)
        if method[0:3].lower()=='pct':
            try:
                pctval = int(method[3:5])
            except TypeError:
                pctval = 5
            self._percentile_discriminator(pct=pctval)
        elif method=='kmeans':
            self._k_means_discriminator()
        
        #return self._clusters_to_time_int()
        self.tst, self.tend = self._clusters_to_time_int(min_int=min_len,
                                                         max_int=max_len)
        
    def _calc_amplitude(self,nwind=4096):
        '''
        calculates amplitude for amplitude discriminator
        '''
        self.nfr = int(nwind/2)
        self.at, ampl = rmsWind(self.x,sr=self.sr,nwind=self.nwind,
                                    nhop = self.nfr)
        self.ax = 20*np.log10(ampl)

    def _calc_band_amplitude(self,nwind=4096,fmin=50,fmax=5000):
        '''
        calculates amplitude in a frequency band for amplitude discriminator
        '''
        from ..FFTFilters import FilterBank, PiecewiseFilterSpec
        self.nfr = int(nwind/2)
        if fmin is None:
            fb = FilterBank([PiecewiseFilterSpec(freq=fmax,mode='lp',sr=self.sr)],
                            sr=self.sr,nwind=self.nwind,nhop=self.nfr)

        elif fmax is None:
            fb = FilterBank([PiecewiseFilterSpec(freq=fmin,mode='hp',sr=self.sr)],
                            sr=self.sr,nwind=self.nwind,nhop=self.nfr)
        else:
            fb = FilterBank([PiecewiseFilterSpec(freq=[fmin,fmax],mode='bp',sr=self.sr)],
                            sr=self.sr,nwind=self.nwind,nhop=self.nfr)
        
        ampl, self.at = fb.specout(self.x)
        self.ax = 20*np.log10(ampl.flatten())
        
    def _k_means_discriminator(self, batch_size=45):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics.pairwise import pairwise_distances_argmin
        
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=2, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
        #t0 = time.time()
        X = np.log10(self.ax.reshape(-1, 1))
        mbk.fit(X)
        cc = np.sort(mbk.cluster_centers_,axis=0)
        self.clusters = pairwise_distances_argmin(X,cc)
        

        
    def _percentile_discriminator(self, pct=5):
        '''
        calculate threshold based on percentiles
        
        arguments:
        pct: percentile value
        '''
        self.amin = np.percentile(self.ax, pct)
        self.amax = np.percentile(self.ax, 100-pct)
        self.ath  = (self.amax+self.amin)/2
        
        self.clusters = np.zeros(self.ax.shape[0],dtype='int')
        self.clusters[self.ax>self.ath]=1

    def _clusters_to_time_int(self, min_int=0.0, max_int=None):
        
        tfr = self.nfr/float(self.sr)
        min_frames = int(np.round(min_int/tfr))
        if max_int:
            maxlen = int(max_int/tfr)
        else:
            maxlen = len(self.ax)
        
        lastsplit=0
        
        i=0
        
        off=True
        
        nframes = len(self.clusters)
        
        tst=[]
        tend=[]
        
        while i+min_frames<nframes:
            if self.clusters[i]>0 and off:
                tst.append(self.at[max(0,i-1)])
                off=False
                i+=1
            elif self.clusters[i]<=0 and not off:
                if all(self.clusters[i:i+min_frames] <=0 ):
                    off=True
                    tend.append(self.at[i])
                    i+=min_frames
                else:
                    i+=1
            else:
                i+=1
        if not off:
            tend.append(self.at[-1])
        return tst,tend
    
    def to_textgrid(self, filename='segmentation.TextGrid', 
                    tiername='Segmentation'):
        
        from pympi import TextGrid
        tg=TextGrid(xmax=max(self.tend))
        tier=tg.add_tier(tiername)
        for ii, (ts,te,lab) in enumerate(zip(self.tst,self.tend,self.label)):
            tier.add_interval(ts,te,'{}'.format(lab))
        
        tg.to_file(filename)

    def output(self, file_handle):
        for ii, (ts,te,lab) in enumerate(zip(self.tst,self.tend,self.label)):
            file_handle.write('{},{},{}\n'.format(ts,te,lab))

    def recognise(self, mode='sphinx', marg=0.2):
        import speech_recognition as srec
        # use the audio file as the audio source
        r = srec.Recognizer()
        
        if mode=='sphinx':
            recogniser = r.recognize_sphinx
            sys.stderr.write('Doing speech recognition with sphinx\n')
        if mode=='google':
            sys.stderr.write('Doing speech recognition with google\n')
            recogniser = r.recognise_google
        
        for ii, (ts,te,lab) in enumerate(zip(self.tst,self.tend,self.label)):
            tstart = ts-marg
            tend = te+marg
            wo=self.x[int(tstart*self.sr):int(tend*self.sr)]
    
            wavwrite('speech_sample.wav',self.sr,wo.astype('int16'))
    
            with srec.AudioFile('speech_sample.wav') as source:
                audio = r.record(source)  # read the entire audio file
    
            try:
                # for testing purposes, we're just using the default API key
                # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                # instead of `r.recognize_google(audio)`
                utt = recogniser(audio)
                #utt = r.recognize_sphinx(audio)
                self.label[ii] = utt
                sys.stderr.write('{}\n'.format(utt))
            except srec.UnknownValueError:
                sys.stderr.write("Speech Recognition could not understand audio\n")
            except srec.RequestError as e:
                sys.stderr.write("Could not request results {}\n".format(e))

class MultiChannelSegmenter(object):
    '''
    Detects regions from different sources from multi-channel recordings
    '''
    
    def __init__(self, x, sr=1, nwind=4096, method = 'kmeans',
                 nsources = 2, min_len = 0.1, max_len=5):
        '''
        crate multi-channel analyser
        '''
        self.x=x
        self.sr=sr
        self.nwind=nwind
        self.nsources = nsources
        
        self._calc_amplitude(nwind=nwind)
        if method[0:3].lower()=='pct':
            try:
                pctval = int(method[3:5])
            except TypeError:
                pctval = 5
            self._percentile_discriminator(pct=pctval)
        elif method=='kmeans':
            self._k_means_discriminator()
        
        #return self._clusters_to_time_int()
        self.tst, self.tend, self.label = self._clusters_to_time_int(
                                                        min_int=min_len,
                                                        max_int=max_len)
        
    def _calc_amplitude(self,nwind=4096):
        '''
        calculates amplitude for amplitude discriminator
        '''
        self.nfr = int(nwind/2)
        
        ax=[]
        for i in range(self.x.shape[1]):
            self.at, axi = rmsWind(self.x[:,i],sr=self.sr,
                                  nwind=self.nwind,
                                  nhop = self.nfr)
            ax.append(axi)
        self.ax = np.array(ax).T
        self.dt = self.at[1]-self.at[0]
        
    def _k_means_discriminator(self, batch_size=45):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics.pairwise import pairwise_distances_argmin
        
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=self.nsources+1, 
                              batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
        #t0 = time.time()
        X = np.log10(self.ax)
        mbk.fit(X)
        cc = np.zeros(mbk.cluster_centers_.shape)
        # index of cluster corresponding to silence
        idx_silence = np.argmin(np.sum(mbk.cluster_centers_,axis=1))
        cc[0,:] = mbk.cluster_centers_[idx_silence,:]
        idx_free = range(cc.shape[0])
        idx_free.remove(idx_silence)
        cred = mbk.cluster_centers_-cc[0,:]
        # remaining indexes, sort them by channel
        used_chan=[]
        nchan = cc.shape[1]
        last_unmatched=0
        while idx_free:
            crem = cred[idx_free,:]
            r,idx_chan = np.unravel_index(crem.argmax(),crem.shape)
            idx_center = idx_free[r]
            if idx_chan not in used_chan:
                this_center = idx_chan+1
            else:
                # append to end of list
                this_center = cc.shape[0]-last_unmatched-1
                sys.stderr.write('Cluster {} not matched to channel\n'.format(idx_center))
            cc[this_center,:]=mbk.cluster_centers_[idx_center,:]
            used_chan.append(idx_chan)
            idx_free.remove(idx_center)
        
        cc[1:,:] = np.delete(mbk.cluster_centers_,idx_silence,axis=0)
        #cc = mbk.cluster_centers_[idxs,:]
        self.clusters = pairwise_distances_argmin(X,cc)
        self.centers = cc
        

    def _clusters_to_time_int(self, min_int=0.0, max_int=None):
        
        tfr = self.nfr/float(self.sr)
        min_frames = int(np.round(min_int/tfr))
        if max_int:
            maxlen = int(max_int/tfr)
        else:
            maxlen = len(self.ax)
        
        lastsplit=0
        
        i=0
        
        off=True
        
        nframes = len(self.clusters)
        
        tst=[]
        tend=[]
        label=[]
        
        lastlabel = 0
        
        while i+min_frames<nframes:
            if self.clusters[i]>0 and off:
                tst.append(self.at[max(0,i-1)])
                label.append(self.clusters[i])
                lastlabel = self.clusters[i]
                off=False
                i+=1
            elif self.clusters[i]<=0 and not off:
                if all(self.clusters[i:i+min_frames] <=0 ):
                    off=True
                    tend.append(self.at[i])
                    i+=min_frames
                else:
                    i+=1

            elif self.clusters[i] != lastlabel and not off:
                tend.append(self.at[i-1])#-self.dt/2)
                tst.append(self.at[i-1])
                label.append(self.clusters[i])
                lastlabel = self.clusters[i]
                off=False
                i+=1

            else:
                i+=1
        if not off:
            tend.append(self.at[-1])
        return tst,tend,label
    
    def to_textgrid(self, filename='mc_segmentation.TextGrid', 
                    tiername='Segmentation'):
        
        from pympi import TextGrid
        tg=TextGrid(xmax=max(self.tend))
        tier=tg.add_tier(tiername)
        for ii, (ts,te,lab) in enumerate(zip(self.tst,self.tend,self.label)):
            tier.add_interval(ts,te,'{}'.format(lab))
        
        tg.to_file(filename)
        
    def output(self, file_handle):
        for ii, (ts,te,lab) in enumerate(zip(self.tst,self.tend,self.label)):
            filehandle.write('{},{},{}'.format(ts,te,lab))
            
    def recognise(self, mode='sphinx', marg=0.2):
        # use the audio file as the audio source
        r = srec.Recognizer()
        
        if mode=='sphinx':
            recogniser = r.recognize_sphinx
        if mode=='google':
            recogniser = r.recognise_google
        
        for ii, (ts,te,lab) in enumerate(zip(self.tst,self.tend,self.label)):
            tstart = st-marg
            tend = end+marg
            wo=w[int(tstart*fs):int(tend*fs)]
    
            wavwrite('speech_sample.wav',fs,wo)
    
            with srec.AudioFile('speech_sample.wav') as source:
                audio = r.record(source)  # read the entire audio file
    
            try:
                # for testing purposes, we're just using the default API key
                # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                # instead of `r.recognize_google(audio)`
                utt = recogniser(audio)
                #utt = r.recognize_sphinx(audio)
                self.label[ii] = utt
            except srec.UnknownValueError:
                print("Speech Recognition could not understand audio")
            except srec.RequestError as e:
                print("Could not request results {}".format(e))
        
def output_results(seg, output_csv='', output_text_grid=''):
    if output_text_grid:
        seg.to_textgrid(output_text_grid)

    if output_csv:
        with open(output_csv,'w') as f:
            seg.output(f)
    else:
        seg.output(sys.stdout)


def analyse_rec(sound_files, nsources=1, wind_sec=0.092, min_len=.3,
                recognise=None, output_csv='', output_text_grid=''):
    # segment recordings
    w=[]
    for ff in sound_files:
        sr,wi=wavread(ff)
        w.append(wi.T)
    
    w=np.vstack(w).T
    sys.stderr.write("Read {} files, {} channels, {} samples\n"\
                     .format(len(sound_files),w.shape[1],w.shape[0]))
    sys.stderr.write("Segmenting audio\n")
    if nsources>1:
        seg = MultiChannelSegmenter(w,sr=sr,min_len=min_len)
    else:
        #w=w.squeeze()
        if len(w.shape)>1:
            w = np.mean(w,axis=1)
        seg = SilenceDetector(w.squeeze(), sr=sr, method = 'pct05',
                                min_len=min_len, wind_sec=wind_sec)
        seg.label = [1 for tst in seg.tst]
        seg.centers = np.array([[0,0],[1,0]])
        
    if recognise:
        seg.recognise(mode=recognise)

                                
            
    sys.stderr.write("Found {} chunks\n".format(len(seg.label)))
    
    output_results(seg, output_csv=output_csv, 
                        output_text_grid=output_text_grid)

def process_file_list(batch_file, output_csv='', 
                                  output_text_grid='',
                                  recognise=None, 
                                  wind_sec=0.092, 
                                  min_len=.3, 
                                  nsources=0):
                                  
    import logging
    file_seq=[]
    
    suffix_csv = output_csv
    suffix_tg = output_text_grid
    out_csv=''
    out_tg=''
    
    if not (suffix_csv or suffix_tg):
        suffix_csv = '_segmentation.csv'
    
    with open(batch_file) as f:
        for line in f:
            files = [it.strip() for it in line.split(',') if len(it.strip())>0]
            
            if len(files)>0:
                basedir, filename = os.path.split(files[0])
                if suffix_csv:
                    out_csv,ext = os.path.splitext(files[0])
                    out_csv+=suffix_csv
                if suffix_tg:
                    out_tg,ext = os.path.splitext(files[0])
                    out_tg+=suffix_tg
                try:
                    analyse_rec(files, output_csv=out_csv, 
                                output_text_grid=out_tg,
                                nsources=len(files),
                                recognise=recognise,
                                wind_sec=wind_sec,
                                min_len=min_len)
                except Exception as e:
                    message = 'ERROR while processing files:\n'
                    for f in files:
                        message+=f
                    message+='/n'
                    logging.exception(message)
                    #~ sys.stderr.write('ERROR while processing files:\n')
                    #~ for f in files:
                        #~ sys.stderr.write(f+'\n')
                    #~ sys.stderr.write(str(e))
                    #~ sys.stderr.write('\n')
                    #~ sys.stderr.write(e.__doc__ )
                    #~ sys.stderr.write('\n')
    return 0


def main(args):
    sound_files = args.infiles
    for sf in sound_files:
        sys.stderr.write(sf+', ')
    
    sys.stderr.write('\n')
    
    if args.batch:
        process_file_list(args.batch, nsources=args.n_sources, 
                                     wind_sec=args.window,
                                     min_len=args.min_silence,
                                     output_csv=args.csv,
                                     output_text_grid=args.textgrid,
                                     recognise=args.recognise)
        
    else:
        if sound_files:  
            analyse_rec(sound_files, nsources=args.n_sources, 
                                     wind_sec=args.window,
                                     min_len=args.min_silence,
                                     output_csv=args.csv,
                                     output_text_grid=args.textgrid,
                                     recognise=args.recognise)
        else:
            sys.stderr.write('Input files or batch list (-b) are required!\n')
        

    return 0


if __name__ == '__main__':
    import sys
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--min-silence", nargs='?', default = '0.3', type=float,
        help = "minimum silence duration in seconds")
    ap.add_argument("-w", "--window", nargs='?', default = '0.092', type=float,
        help = "window analysis duration in seconds")
    ap.add_argument("-b", "--batch", nargs='?', 
        help = "input file list for batch processing")
    ap.add_argument("-r", "--recognise", nargs='?', 
        help = "use speach recognition on each interval. Select method sphinx or google")
    ap.add_argument("-c", "--csv", nargs='?', default = '',
        help = "output to csv file name")
    ap.add_argument("-t", "--textgrid", nargs='?', default = '',
        help = "output to Praat Textgrid file name")
    


    ap.add_argument("-s", "--n-sources", type=float, nargs='?', default = '1',
        help = "number of expected sources in the file")
    
    
    ap.add_argument('infiles', nargs='*', help='Input sound files (required if not batch)')
    
    args = ap.parse_args()

    

    sys.exit(main(args))

