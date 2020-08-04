import argparse
import os
from glob import glob

import numpy as np
from scipy.io import wavfile

import pypevoc.FFTFilters as ft


def read_wav_file(sndfile):
    return wavfile.read(sndfile)

def melspec(sr,w,twind=0.025,thop=0.01,mfcc=False):
    mfb = ft.MelFilterBank(sr=sr,twind=twind, thop=thop)
    wp = ft.preemph(w,hpFreq=50,Fs=sr)
    cs,ms,ts = mfb.mfcc_and_mel(wp)
    return cs,np.log(ms),ts


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', help='Input file or dir')
    parser.add_argument('--window_sec', '-w', help='window duration in seconds', default=0.025, type=float)
    parser.add_argument('--hop_sec','-H', help='hop in seconds', default=0.01, type=float)
    return parser.parse_args() 
    
def process_file(sndfile, twind=0.025, thop=0.01, output=None):
    sr,w = read_wav_file(sndfile)
    mc, ms, tm = melspec(sr,w,twind=twind,thop=thop)
    if output is None:
        basename = os.path.splitext(sndfile)[0]
        dname = basename+'_MEL_MFCC.npz'
    else:
        dname = output
    np.savez(dname, mfcc=mc, melspec=ms, t=tm)

def process_dir(directory):
    from glob import glob
    filelist = glob(os.path.join(directory,'*.wav'))
    for sndfile in filelist:
        process_file(sndfile)

if __name__ == '__main__':
    args = parse_args()
    if os.path.isdir(args.input):
        datadir = args.input
        process_dir(datadir)
    else:
        process_file(args.input, twind=args.window_sec, thop=args.hop_sec)