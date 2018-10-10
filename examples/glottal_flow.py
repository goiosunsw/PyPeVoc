import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy.io import wavfile

from pypevoc.speech.glottal import iaif_ola

try:
    filename = sys.argv[1]
except IndexError:
    filename = "hide.wav"

sr, w = wavfile.read(filename)

g, dg, vt, gf = iaif_ola(w, Fs=sr)

t = np.arange(len(w))/sr

fig,ax = pl.subplots(2,sharex=True)
ax[0].plot(t,w)
ax[1].plot(t,g)

try: 
    import matlab.engine
    import matlab
    eng = matlab.engine.start_matlab()
except ImportError:
    pass
else:
    eng.addpath(eng.genpath('~/Devel/covarep/'))
    try:
        g_m, dg_m, vt_m, gf_m = eng.iaif_ola(matlab.double(w.tolist()),
                                     float(sr),
                                     nargout=4)
    except matlab.engine.MatlabExecutionError:
        pass
    else:
        ax[1].plot(t,np.array(g_m).flatten())
    finally:
        eng.quit()

pl.show()
