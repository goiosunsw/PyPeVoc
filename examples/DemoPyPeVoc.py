import numpy  as np
import pylab  as pl
import pandas as pd
import sys
sys.path.append('..')
import PVAnalysis as pv

# Sample rate
sr = 44100
#sr=150000

# Vibrato frequency
vibfreq = 5.0

# Average amplitude of harmonics
hamp0 = 0.1*np.array([1, .5, .3])
#hamp0 = 0.1*np.array([0.9])

# Fraction variation of harmonics in vibrato
hvib = 1.0*np.array([.5,0.1,.9])
#hvib = 1.0*np.array([-.2])
# relative phase of harmonic variation
hph = np.array([0,np.pi/2,np.pi])
# Mean fundamental frequency
f0 = 500

# Depth of frequency vibrato
f0vib = 0.01

# signal duration
dur = 1.0

sig = np.zeros(int(sr*dur)) + 0.00*(np.random.rand(int(sr*dur))-.5)
vibsig = np.zeros(int(sr*dur))
hvibsig = np.zeros((int(sr*dur),len(hamp0)))
t = np.arange(0,dur,1./sr)

vibsig = np.sin(2*np.pi*vibfreq*t)

f0sig = f0 * (1 + f0vib*vibsig)

for n,ha in enumerate(hamp0):
    hno = n+1
    fsig = f0sig*hno
    phsig = np.cumsum(2*np.pi*fsig/sr)
    hvibsig[:,n] = ha * (1+hvib[n]*np.sin(2*np.pi*vibfreq*t+hph[n]))
    sig += (hvibsig[:,n]) * np.sin(phsig)
    
#pl.plot(sig)
pl.figure()
ss=pl.specgram(sig,NFFT=1024/2)

mypv=pv.PV(sig,sr,nfft=1024,npks=len(hamp0))
mypv.run_pv()
mypv.plot_time_freq()

ss=mypv.toSinSum()
ss.plot_time_freq_mag(minlen=5)

w=ss.synth(sr,mypv.hop/1)

#pl.hold(True)
pl.plot(sig,label='orig')
pl.hold(True)
pl.plot(w,label='resynth')
pl.legend()
