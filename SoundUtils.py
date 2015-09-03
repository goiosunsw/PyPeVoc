import numpy as np


def RMSWind(x, sr=1, nwind=1024, nhop=512, windfunc=np.blackman):
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
    #wsum = sum(wind)
    wsum2= sum(wind*wind)
    
    while (iend < nsam):
        thisx = x[ist:iend]
        xw = thisx*wind
        
        amp.append(np.sqrt(sum(xw*xw)/wsum2))
        t.append(float(ist+iend)/2.0/float(sr))
        
        ist = ist+nhop
        iend = ist+nwind
        
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