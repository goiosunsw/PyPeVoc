import numpy as np
from pypevoc.speech.SpeechAnalysis import Formants
from pypevoc.SoundUtils import RMSWind
from pypevoc.PVAnalysis import PV
from pypevoc.Heterodyne import HeterodyneHarmonic

formant_window = .05

def get_formants(w,sr, twind=formant_window):
    t, f, bw = Formants(w, sr, tWind=twind)
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
    