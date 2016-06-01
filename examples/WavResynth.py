import numpy  as np
import pylab  as pl
import pandas as pd
import sys
from scipy.io import wavfile as wf

sys.path.append('..')
import PVAnalysis as pv

#sr, sig =  wf.read('pepperCl.wav')
#sr, sig =  wf.read('pepperSx.wav')
#sr, sig =  wf.read('perlmanVn.wav')
sr, sig =  wf.read('smirnoffVn.wav')
sig = sig/ float(np.iinfo(sig.dtype).max)
    
#pl.plot(sig)
pl.figure()
ss=pl.specgram(sig,NFFT=1024/2)

mypv=pv.PV(sig,sr,nfft=1024,npks=25,hop=256)
mypv.run_pv()
mypv.plot_time_freq()

ss=mypv.toSinSum()
#ss.plot_time_freq_mag(minlen=5)

w=ss.synth(sr,mypv.hop/1)

#pl.hold(True)
pl.figure()
pl.plot(sig,label='orig')
pl.hold(True)
pl.plot(w,label='resynth')
pl.legend()
pl.show()
