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
    
def PlaySound(w,sr=44100):
    import pyaudio
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1, rate=sr, output=1)


    stream.write(w.astype(np.float32).tostring())
    
    stream.close()
    p.terminate()
    