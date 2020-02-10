import numpy as np
from pypevoc.speech.SpeechAnalysis import Formants
from pypevoc.SoundUtils import RMSWind
from pypevoc.PVAnalysis import PV
from pypevoc.Heterodyne import HeterodyneHarmonic

from phoneme_segmenter import *

formant_window = .05

def get_formants(w,sr, twind=formant_window):
    t, f, bw = Formants(w.copy(), sr, tWind=twind)
    fmed = np.nanmedian(f,axis=0)
    ret = dict()
    for ii, ff in enumerate(fmed):
        ret['F{}'.format(ii+1)] = ff
    return ret

def get_RMS(w,sr):
    t,a=RMSWind(w,sr)
    return {'RMS':np.nanmean(a)}

def get_f0(w,sr, nfft=2048,pkthresh=1e-8,npks=50):
    pv = PV(w,sr,nfft=nfft,pkthresh=pkthresh,npks=npks)
    pv.run_pv()
    f0 = pv.calc_f0(thr=0.01)
    
    nhop = 1024
    t,a=RMSWind(w,sr,nwind=nfft,nhop=nhop)
    hh=HeterodyneHarmonic(w,sr,f=np.nanmean(f0))
    f0,tf0=hh.calc_adjusted_freq(hh.f0)
    hh=HeterodyneHarmonic(w,sr,tf=tf0,f=f0,nharm=20,nwind=nfft,nhop=nhop)

    hpct = np.sqrt(np.sum(np.abs(hh.camp)**2,axis=1))/a
    
    return {'F0':np.nanmean(f0),
            'RMS':np.nanmean(a),
            'Harmonicity':np.nanmedian(hpct)}



def describe_phoneme(w,sr):
    desc = {}
    desc.update(get_f0(w,sr))
    #desc.update(get_RMS(w,sr))
    desc.update(get_formants(w,sr))
    return desc
    

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', help='Input file or dir')
    return parser.parse_args() 

def output_interval(w,sr,ts,te,label):
    try:
        desc = describe_phoneme(w,sr)
    except Exception:
        desc = {}
    dstr = '{:.3f}, {:.3f}, {}'.format(ts,te,label)
    for k,v in desc.items():
        dstr+=',{}'.format(v)
    print(dstr)
    
def process_file(sndfile):
    sr,w = read_wav_file(sndfile)
    ints = file_segments(sr,w)
    tps = 0.
    for thisi in ints:
        ts = thisi['start']
        te = thisi['end']
        tph = thisi['phonemes']
        tpe = ts
        ww = w[int(sr*tps):int(sr*tpe)]
        tps = tpe
        output_interval(ww,sr,ts,te,'Utteration START')
        for tpe in tph[:-1]:
            ww = w[int(sr*tps):int(sr*tpe)]
            output_interval(ww,sr,ts,te,'phoneme')
            tps=tpe
        tpe = te
        ww = w[int(sr*tps):int(sr*tpe)]
        output_interval(ww,sr,ts,te,'Utteration END')


if __name__ == '__main__':
    args = parse_args()
    # if os.path.isdir(args.input):
    #     datadir = args.input
    #     process_dir(datadir)
    # else:
    process_file(args.input)