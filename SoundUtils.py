import numpy as np


def FuncWind(func, x, sr=1, nwind=1024, nhop=512, power=1, windfunc=np.blackman):
    '''
    Applies a function window by window to a time series
    '''
    
    nsam = len(x)
    ist = 0
    iend = ist+nwind

    t=[]
    ret=[]
    
    wind = windfunc(nwind)
    if power > 0:
        wsumpow = sum(wind**power)
    else:
        wsumpow = 1.
    
    while (iend < nsam):
        thisx = x[ist:iend]
        xw = thisx*wind

        ret.append(func(xw)/wsumpow)
        t.append(float(ist+iend)/2.0/float(sr))
        
        ist = ist+nhop
        iend = ist+nwind
        
    return np.array(ret), np.array(t)
    

def RMSWind(x, sr=1, nwind=1024, nhop=512, windfunc=np.blackman):
    '''
    Calculates the RMS amplitude amplitude of x, in frames of
    length nwind, and in steps of nhop. windfunc is used as 
    windowing function.
    
    nwind should be at least 3 periods if the signal is periodic.  
    '''
    
    
    def RMSvec(xw):
        return sum(xw*xw)
    
    amp, t = FuncWind(RMSvec,x,power=1,sr=sr,nwind=nwind,nhop=nhop)
    
    return np.sqrt(np.array(amp)), np.array(t)
  
def Heterodyn(x, f, sr=1, nwind=1024, nhop=512,windfunc=np.blackman):  
    '''
    Calculates the power near frequency f in x
    
    nwind should be at least 3 periods if the signal is periodic.  
    '''
    sinsig = np.exp(2j*np.pi*np.arange(len(x))*f/float(sr))
    hamp,t=FuncWind(np.sum, x*sinsig, power=1, sr=sr,nwind=nwind,nhop=nhop)
    return np.array(hamp), np.array(t)


def SpecCentWind(x, sr=1, nwind=1024, nhop=512, windfunc=np.blackman):
    '''
    Calculates the SpectralCentroid of x, in frames of
    length nwind, and in steps of nhop. windfunc is used as 
    windowing function.
    
    nwind should be at least 3 periods if the signal is periodic.  
    '''
    ff = np.arange(nwind/2)/float(nwind)*sr
    
    def SCvec(xw):
        xf = np.fft.fft(xw)
        xf2 = xf[:nwind/2]
        return sum(np.abs(xf2)*ff)/sum(np.abs(xf2))
    
    amp, t = FuncWind(SCvec,x,power=0,sr=sr,nwind=nwind,nhop=nhop)
    
    return np.array(amp), np.array(t)
    

def AvgWind(x, sr=1, nwind=1024, nhop=512, windfunc=np.blackman):
    '''
    Calculates the RMS amplitude amplitude of x, in frames of
    length nwind, and in steps of nhop. windfunc is used as 
    windowing function.
    
    nwind should be at least 3 periods if the signal is periodic.  
    '''
    
    nsam = len(x)
    ist = 0
    iend = ist+nwind

    t=[]
    amp=[]
    
    wind = windfunc(nwind)
    wsum = sum(wind)
    #wsum2= sum(wind*wind)
    
    while (iend < nsam):
        thisx = x[ist:iend]
        xw = thisx*wind
        
        amp.append(sum(xw)/wsum)
        t.append(float(ist+iend)/2.0/float(sr))
        
        ist = ist+nhop
        iend = ist+nwind
        
    return np.array(amp), np.array(t)

def SpecFlux(x, sr=1, nwind=1024, nhop=512, minf=0, maxf=np.inf, windfunc=np.blackman):
    '''
    Calculates the spectral flux in sunud
    '''
    
    nsam = len(x)
    # first window
    ist = 0
    iend = ist+nwind

    t=[]
    res=[]
    
    wind = windfunc(nwind)
    wsum = sum(wind)
    #wsum2= sum(wind*wind)
    
    minbin = int(minf/sr*nwind)
    maxbinf = (float(maxf)/sr*nwind)
    if maxbinf > nwind:
        maxbin = nwind
    else:
        maxbin = int(maxbinf)
    
    while (iend < nsam-nhop):
        thisx = x[ist:iend]
        nextx = x[ist+nhop:iend+nhop]
        
        ff = np.abs(np.fft.fft(thisx*wind))
        fl = np.abs(np.fft.fft(nextx*wind))
        
        res.append(np.sqrt(sum((ff[minbin:maxbin]-fl[minbin:maxbin])**2)))
        t.append(float(ist+iend+nhop)/2.0/float(sr))
        
        ist = ist+nhop
        iend = ist+nwind
        
    return np.array(res), np.array(t)

def aubio_f0yin(y,sr, nwind=1024, hop=512, method='yin', tolerance=None):
    ''' Applies f0 detection to a numpy vector using aubio
    '''
    from aubio import pitch,fvec
    
    po = pitch(method, nwind, hop, sr)
    vs = fvec(nwind)
    
    if tolerance is not None:
        if tolerance>0.0 and tolerance < 1.0:
            po.set_tolerance(tolerance)
        else:
            sys.stderr.write('Tolerance not set: Out of bounds\n')
    
    nsamples = y.shape[0]
    
    freq=[]
    time=[]
    conf=[]
    
    for ii in xrange(0,nsamples-nwind,hop):
        thisy = y[ii:ii+nwind]
        vs[:] = thisy
        time.append(float(ii+nwind/2)/sr)
        freq.append(po(vs))
        conf.append(po.get_confidence())
    return np.array(freq).squeeze(), np.array(time), np.array(conf)
    
def PlaySound(w,sr=44100):
    import pyaudio
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1, rate=sr, output=1)


    stream.write(w.astype(np.float32).tostring())
    
    stream.close()
    p.terminate()
    
