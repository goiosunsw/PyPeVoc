import sys
import numpy as np
import traceback
import pandas
from pypevoc.speech.SpeechAnalysis import Formants
from pypevoc.SoundUtils import RMSWind
from pypevoc.PVAnalysis import PV
from pypevoc.Heterodyne import HeterodyneHarmonic

from phoneme_segmenter import *


formant_window = .05
moment_window=.01

def get_formants(w,sr, twind=formant_window, twind_min=0.01):
    while twind>=twind_min:
        try:
            t, f, bw = Formants(w.copy(), sr, tWind=twind)
            f[0][0]
        except (ValueError, IndexError):
            twind/=2
            f=[[]]
            continue
        break
        
    fmed = np.nanmedian(f,axis=0)
    ret = dict()
    for ii, ff in enumerate(fmed):
        ret['F{}'.format(ii+1)] = ff
    return ret

def get_RMS(w,sr):
    a,t=RMSWind(w,sr)
    return {'RMS':np.nanmean(a)}

def get_f0(w,sr, tfft=0.04,pkthresh=1e-8,npks=50, nfftmin=128):
    ret = {'RMS':np.nan,'f0':np.nan,'Harmonicity':np.nan}
    nfft = next_power_2(sr*tfft)
    nhop = nfft//2
    
    while nfft>nfftmin:
        a,t=RMSWind(w,sr,nwind=nfft,nhop=nhop)
        if len(a)<1:
            nfft=nfft//2
            nhop=nfft//2
        else:
            break
    ret['RMS'] = np.nanmean(a) 
    
    try:
        pv = PV(w,sr,nfft=nfft,pkthresh=pkthresh,
            npks=npks,progress=False)
        pv.run_pv()
        f0 = pv.calc_f0(thr=0.01)
        ret['f0']=np.nanmean(f0)
    except Exception:
        sys.stderr.write('Error calculating f0\n')
        return ret

    try:
        hh=HeterodyneHarmonic(w,sr,f=np.nanmean(f0),nwind=nfft,nhop=nhop)
    except Exception:
        sys.stderr.write('Error in first pass of Heterodyne\n')
        return ret
    try:
        f0,tf0=hh.calc_adjusted_freq(hh.f0)
        hh=HeterodyneHarmonic(w,sr,tf=tf0,f=f0,nharm=20,nwind=nfft,nhop=nhop)
    except Exception:
        sys.stderr.write('Error in second pass of Heterodyne\n')

    hpct = np.sqrt(np.sum(np.abs(hh.camp)**2,axis=1))/a
    
    ret['Harmonicity'] = np.nanmedian(hpct)
    return ret

def get_spectral_moments(w,sr,tfft=moment_window):
    nfft = next_power_2(sr*tfft)
    wo = w.copy()
    wo[:-1] -= wo[1:]
    
    fsg, tsg, sg = sig.spectrogram(wo, fs=sr, nfft=nfft)
    avs = np.mean(sg,axis=1)
    cent = np.sum(avs*fsg)/np.sum(avs)
    var = np.sum(avs*(fsg-cent)**2/np.sum(avs))
    return {'Centroid': cent,
            'Stdev': np.sqrt(var)}
    

def describe_phoneme(w,sr):
    desc = {}
    try:
        desc.update(get_f0(w,sr))
    except Exception:
        traceback.print_exc()
    #desc.update(get_RMS(w,sr))
    try:
        desc.update(get_spectral_moments(w,sr))
    except Exception:
        traceback.print_exc()
    
    try:
        desc.update(get_formants(w,sr))
    except Exception:
        traceback.print_exc()
    return desc
    

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', help='Input file or dir')
    return parser.parse_args() 

def output_interval(w,sr,ts,te,label):
    try:
        desc = describe_phoneme(w,sr)
    except Exception:
        traceback.print_exc()
        desc = {}
    dstr = '{:.3f}, {:.3f}, {}'.format(ts,te,label)
    for k,v in list(desc.items()):
        dstr+=',{}'.format(v)
    print(dstr)

def dict_interval(w,sr,ts,te,label):
    try:
        desc = describe_phoneme(w,sr)
    except Exception:
        traceback.print_exc()
        desc = {}
    return desc

def file_df(sndfile):
    sr,w = read_wav_file(sndfile)
    ints = file_segments(sr,w)
    tps = 0.
    tpe=0.
    alld = []
    for thisi in ints:
        ts = thisi['start']
        te = thisi['end']
        tph = thisi['phonemes']
        label = 'SILENCE'
        tps = tpe
        tpe = ts
        ww = w[int(sr*tps):int(sr*tpe)]
        alld.append(dict_interval(ww,sr,tps,tpe,label))
        alld[-1].update({'t_start':tps,
                            't_end':tpe,
                            'label':label})
        
        tps = ts
        tpe = tps
        label = 'Utteration START'
        for tper in tph:
            tpe = tper+ts
            ww = w[int(sr*tps):int(sr*tpe)]
            alld.append(dict_interval(ww,sr,tps,tpe,label))
            alld[-1].update({'t_start':tps,
                             't_end':tpe,
                             'label':label})
            tps=tpe
            label = 'phoneme'
        tps = tpe
        tpe = te
        label = 'Utteration END'
        ww = w[int(sr*tps):int(sr*tpe)]
        alld.append(dict_interval(ww,sr,tps,tpe,label))
        alld[-1].update({'t_start':tps,
                            't_end':tpe,
                            'label':label})
    df = pandas.DataFrame(alld)
    return df
    
def process_file(sndfile):
    df = file_df(sndfile)
    df.to_csv(sys.stdout)

def process_dir(directory):
    from glob import glob
    filelist = glob(os.path.join(directory,'*.wav'))
    for sndfile in filelist:
        df = file_df(sndfile)
        basepath, ext = os.path.splitext(sndfile)
        df.to_csv(basepath+'.csv')
 
if __name__ == '__main__':
    args = parse_args()
    if os.path.isdir(args.input):
        datadir = args.input
        process_dir(datadir)
    else:
        process_file(args.input)