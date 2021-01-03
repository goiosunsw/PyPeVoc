import numpy  as np
import pylab  as pl
import pandas as pd
import sys
from scipy.io import wavfile as wf

# sys.path.append('..')
from pypevoc import PVAnalysis as pv

#sr, sig =  wf.read('pepperCl.wav')
# sr, sig =  wf.read('pepperSx.wav')
#sr, sig =  wf.read('perlmanVn.wav')
#sr, sig =  wf.read('smirnoffVn.wav')
sr, sig =  wf.read('ProtectMarraigeInAmerica.wav')
#sr, sig =  wf.read('SoloGuitarArpegi.wav')

# scale to floating point (range -1 to 1)
sig = sig/ float(np.iinfo(sig.dtype).max)
    
#pl.plot(sig)
pl.figure()
ss=pl.specgram(sig,NFFT=1024//2)

# Build the phase vocoder object
mypv=pv.PV(sig,sr,nfft=1024*4,npks=25*4,hop=256*4)
# Run the PV calculation
mypv.run_pv()
# plot the peaks that were found
mypv.plot_time_freq()

# convert to sinusoidal lines
ss=mypv.toSinSum()

# resynthesise based on PV analysis
# (reduce hop to slow down, increase to accelerate)
w=ss.synth(sr,mypv.hop/1)

# plot original and resynthesis
pl.figure()
pl.plot(sig,label='orig')
# pl.hold(True)
pl.plot(w,label='resynth')
pl.legend()
pl.show()

fig,ax=pl.subplots(2,1,sharex=True)
ax[0].plot(np.arange(len(sig))/float(sr),sig,label='orig')
# ax[0].hold(True)
ax[0].plot(np.arange(len(w))/float(sr),w,label='resynth')
ax[0].legend()
mypv.plot_time_freq(ax=ax[1])

