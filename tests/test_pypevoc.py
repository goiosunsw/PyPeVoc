import PVAnalysis as pv
import numpy as np

sr = 44100
t = np.arange(sr)/float(sr)

f = [400.,1200.]
mag = [.1,.05]
fmul=1.0

minmag = min(mag)*0.001
#minmag=-1

xx = np.zeros(len(t))
for ff,mm in zip(f,mag):
    xx += mm*np.sin(2.0*np.pi*ff*fmul*t)

p=pv.PV(xx,sr,nfft=2**10,hop=2**9)
p.run_pv()
ss=p.toSinSum()


for ii,part in enumerate(ss.partial):
    avmag = np.mean(part.mag)
    if avmag > minmag:
        print 'Partial %d, st=%d, len=%d, f=%f, mag =%f'%(ii,part.start_idx,len(part.f),np.mean(part.f),avmag)