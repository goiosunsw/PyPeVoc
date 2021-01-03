import os
from glob import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.signal as sig

from pypevoc.speech.SpeechChunker import SilenceDetector
from pypevoc.speech.SpeechSegmenter import SpeechSegmenter

bands = [200.,300.,500.,800.,1200.]
segmenter_detect_thresh=.5
chunker_fmin = 100
chunker_fmax = 1000
chunker_wind_sec=.2
chunker_method = 'pct10'

def estimate_noise_background_spcetrum(w, sr, tst, tend, nfft=1024):
    sg, fsg, tsg = sig.specgram(w, fs=sr, NFFT=nfft)
    for ts,te in zip(sd.tend[:-1],sd.tst[1:]):
        isil = (tsg>=ts)&(tsg<=te)
        silence_sg_chunks.append(sg[:,isil])

    silence_sg = np.hstack(silence_sg_chunks)

    return fsg, np.median(silence_sg,axis=1)

def read_wav_file(sndfile):
    return wavfile.read(sndfile)

def segment_wav(w, sr, fmin=100, fmax=1000, wind_sec=.2,method='pct10'):
    sd = SilenceDetector(w,sr=sr,fmin=fmin,fmax=fmax,wind_sec=wind_sec,method=method)
    return sd.tst, sd.tend
    
def next_power_2(x):
    return int(2**np.ceil(np.log2(x)))
    
def phoneme_segment_wav(w,sr, bands=[200.,300.,500.,800.,1200.],
                        detect_thresh=.5,twind=0.04):
    
    nrough = next_power_2(sr*twind)
    ss = SpeechSegmenter(sr=sr, bands=bands,
                         detect_thresh=detect_thresh,
                         rough_window=nrough)
    ss.set_signal(w,sr=sr)
    tph = ss.process(w)
    tph = ss.refine_all_all_bands()
    return tph

def file_segments(sr,w):
    tst, tend = segment_wav(w,sr,fmin=chunker_fmin,fmax=chunker_fmax,
                            wind_sec=chunker_wind_sec,method=chunker_method)
    ints = []
    for ts, te in zip(tst,tend):
        ww = w[int(ts*sr):int(te*sr)]
        tph = phoneme_segment_wav(ww, sr, bands=bands, 
                                  detect_thresh=segmenter_detect_thresh)
        ints.append({'start':ts,
                     'end':te,
                     'phonemes':tph})
    return ints

def process_file(sndfile):
    sr,w = read_wav_file(sndfile)
    ints = file_segments(sr,w)
    for thisi in ints:
        ts = thisi['start']
        te = thisi['end']
        tph = thisi['phonemes']
        print(('{:7.3f}, Speech START'.format(ts)))
        for t in tph:
            print(('{:7.3f}, New phoneme'.format(t+ts)))
        print(('{:7.3f}, Speech END'.format(te)))

def process_dir(directory):
    from glob import glob
    filelist = glob(os.path.join(directory,'*.wav'))
    for sndfile in filelist:
        sr,w = read_wav_file(sndfile)
        ints = file_segments(sr,w)
        for thisi in ints:
            ts = thisi['start']
            te = thisi['end']
            tph = thisi['phonemes']
            print(('{}, {:7.3f}, Speech START'.format(sndfile,ts)))
            for t in tph:
                print(('{}, {:7.3f}, New phoneme'.format(sndfile,t+ts)))
            print(('{}, {:7.3f}, Speech END'.format(sndfile,te)))
        


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', help='Input file or dir')
    return parser.parse_args() 
    

if __name__ == '__main__':
    args = parse_args()
    if os.path.isdir(args.input):
        datadir = args.input
        process_dir(datadir)
    else:
        process_file(args.input)
