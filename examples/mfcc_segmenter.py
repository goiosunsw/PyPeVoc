import os
import argparse
import numpy as np
import pypevoc.FFTFilters as ft
import scipy.signal as sig
from scipy.io import wavfile


def read_wav_file(sndfile):
    return wavfile.read(sndfile)


def mfcc_change_rate(sr,w, twind=0.025, thop=0.01,
                     mode='MELSPEC',ncc=12):
    
    mfb = ft.MelFilterBank(sr=sr,twind=twind, thop=thop)
    wp = ft.preemph(w,hpFreq=50,Fs=sr) 
    if mode == 'MELSPEC':
        feat,tfeat = mfb.specout(wp)
        feat = np.log(feat)
    elif mode == 'MFCC':
        feat, tfeat = mfb.mfcc(wp)
        feat = feat[:,1:ncc+1]
    else:
        raise NotImplementedError, "{} unknown".format(method)

    ndiff = int(np.round(max_tchange/thop))
    dfeat = np.zeros((ndiff,len(tfeat)))
    for ii in range(1,ndiff):
        dfeat[ii,ndiff:-ndiff] = np.sum((feat[:-ndiff*2,:]-feat[ndiff*2:,:])**2,axis=1)
    dfsum = np.sum(dfeat,axis=0)


def mfcc_segments(sr,w,twind=0.025,thop=0.01,
                  max_tchange=0.05,percentile_thresh=50,
                  mode='MELSPEC',
                  ncc=12):
    mfb = ft.MelFilterBank(sr=sr,twind=twind, thop=thop)
    wp = ft.preemph(w,hpFreq=50,Fs=sr) 
    if mode == 'MELSPEC':
        feat,tfeat = mfb.specout(wp)
        feat = np.log(feat)
    elif mode == 'MFCC':
        feat, tfeat = mfb.mfcc(wp)
        feat = feat[:,1:ncc+1]
    else:
        raise NotImplementedError, "{} unknown".format(method)

    ndiff = int(np.round(max_tchange/thop))
    dfeat = np.zeros((ndiff,len(tfeat)))
    for ii in range(1,ndiff):
        dfeat[ii,ndiff:-ndiff] = np.sum((feat[:-ndiff*2,:]-feat[ndiff*2:,:])**2,axis=1)
    dfsum = np.sum(dfeat,axis=0)
    dfspks = sig.argrelmax(dfsum)[0]
    pkthresh = np.percentile(dfsum,percentile_thresh)
    dfspks = dfspks[dfsum[dfspks] > pkthresh]
    return tfeat[dfspks], dfsum[dfspks]


def file_segments(sr,w):
    times, vals = mfcc_segments(sr,w)
    dictlist = []
    for t,v in zip(times,vals):
        dictlist.append({'start':tst,
                         'end':t,
                         'strength':val})    
    import pandas
    return pandas.DataFrame(dictlist)


def process_file(sndfile, mode='MELSPEC'):
    sr,w = read_wav_file(sndfile)
    times, values = mfcc_segments(sr,w,mode=mode)
    for t,v in zip(times, values):
        print("{:f},{:f}".format(t,v))


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
            print('{}, {:7.3f}, Speech START'.format(sndfile,ts))
            for t in tph:
                print('{}, {:7.3f}, New phoneme'.format(sndfile,t+ts))
            print('{}, {:7.3f}, Speech END'.format(sndfile,te))
        


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
        process_file(args.input, mode='MFCC')