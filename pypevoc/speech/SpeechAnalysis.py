import numpy as np
import sys
import scipy.signal as sig
import scipy.linalg as lg
from scipy.io import wavfile
from .. import FFTFilters as ftf
from ..PeakFinder import PeakFinder

def lpc(w, order, axis=-1):
    """
    Calculate the lpc coefficients of the waveform
    """

    nsamp = w.shape[axis]
    if order > nsamp:
        raise ValueError('Order must be smaller than size of vector')

    r = np.correlate(w, w, 'full')
    #use_r = np.zeros(order+1)
    #use_r[:order+1] = r[nsamp-1:nsamp+order]
    use_r = r[nsamp-1:nsamp+order]
    a = lg.solve_toeplitz(use_r[:-1], -use_r[1:])

    return a

def refine_max(x, pos):
    '''
    Given the position of a peak pos in a series x,
    interpolate the position assuming that the peak is
    approximated by a quadratic function
    '''
    if pos==0:
        pos=1

    sur = x[pos-1:pos+2]

    if sur[1]>sur[0] and sur[1]>sur[2]:
        c = sur[1]
        b = (sur[2] - sur[0])/2
        a = (sur[2] + sur[0])/2 - c

        lpos = - b/2/a
        fpos = float(pos) + lpos
        fval = a*lpos*lpos + b*lpos + c
        #print "rpos = %d; rval = %f; val = %f; dpos = %f; pos = %f"%(pos,sur[1],fval, lpos, fpos)

    else:
        fpos = pos
        fval = sur[1]

    return fpos,fval.tolist()


def DistribMoments(x,f, MaxMoments=4):
    '''Calculate the moments in a distribution f(x)
       x: abcissa - values at which distribution is given
       f: value -   values of the distribution
       MaxMoments: maximum moment order to return

       returns:
       COG: center of gravity
       StDev: standard deviation
       skew: skewness
       kurt: kurtosis
       Moments: array with all raw central moments
    '''

    moments = []
    m0 = np.sum(f)
    m1 = np.sum(f*x)/m0
    moments.append(m1)
    for mn in range(1,MaxMoments):
        moments.append(np.sum((x-m1)**(mn+1)*f)/m0)

    cog = m1
    stdev = np.sqrt(moments[1])
    skew = moments[2]/moments[1]**1.5
    kurt = moments[3]/moments[1]**2 - 3.

    return cog,stdev,skew,kurt,moments

def SpectralMoments(w, Fs, tWind=0.025, tHop=0.0125,
                    windFunc=sig.hamming, fCut=300, maxMoments=4):
    '''Calculates spectral moments in short windows  of signal w

        w:          signal
        Fs:         sample rate
        tWind:      window length in seconds
        tHop:       hop length in seconds
        windFunc:   windowing function
        fCut:       high-pass cutoff
        MaxMoments: maximum moment order to return

       returns:
        cog:     center of gravity
        stdev:     standard deviation
        skew:    skewness
        kurt:    kurtosis
        moments: all the moments


    '''

    wLen = len(w)
    hopLenSam = int(np.round(Fs*tHop));
    windowLenSam = int(np.round(Fs*tWind));
    #print 'SpectralMoments: Fs={}; wLen={}; hop={}'.format(Fs,windowLenSam,hopLenSam)
    specLen = int(windowLenSam/2)

    dt = 1./Fs
    nFrames = int((wLen - windowLenSam-1)/hopLenSam)

    wind = windFunc(windowLenSam)
    SxxSum = np.zeros(specLen)
    freqS = (np.arange(specLen))*float(Fs)/windowLenSam

    for FN in np.arange(nFrames):
        I0 = (FN)*hopLenSam;
        Iend = I0 + windowLenSam;
        X = w[I0:Iend];

        XW = X*sig.hamming(len(X));
        XF = np.fft.fft(XW)
        Sxx = np.abs(XF)**2
        SxxSum = SxxSum + Sxx[0:specLen]

    # periodogram
    SxxS = np.sqrt(SxxSum/float(nFrames))

    # intensity
    intens = np.sqrt(np.mean(SxxSum/float(nFrames)))

    # filter out low frequencies
    idx = freqS > fCut

    # compute moments
    cog, std, skew, kurt, mm=DistribMoments(freqS[idx], SxxS[idx], maxMoments)

    return dict(cog=cog, std=std, skew=skew, kurt=kurt, level=intens)

def Periodogram(w, Fs, tWind=0.025, tHop=0.0125,
                    windFunc=sig.hamming):
    '''Calculates spectral moments in short windows  of signal w

        w:          signal
        Fs:         sample rate
        tWind:      window length in seconds
        tHop:       hop length in seconds
        windFunc:   windowing function

       returns:
        Sxx:       power spectrum
        f:         frequency values


    '''

    wLen = len(w)
    hopLenSam = int(np.round(Fs*tHop));
    windowLenSam = int(np.round(Fs*tWind));
    #print 'SpectralMoments: Fs={}; wLen={}; hop={}'.format(Fs,windowLenSam,hopLenSam)
    specLen = windowLenSam/2

    dt = 1./Fs
    nFrames = (wLen - windowLenSam-1)/hopLenSam

    wind = windFunc(windowLenSam)
    SxxSum = np.zeros(specLen)
    freqS = (np.arange(specLen))*float(Fs)/windowLenSam

    for FN in np.arange(nFrames):
        I0 = (FN)*hopLenSam;
        Iend = I0 + windowLenSam;
        X = w[I0:Iend];

        XW = X*sig.hamming(len(X));
        XF = np.fft.fft(XW)
        Sxx = np.abs(XF)**2
        SxxSum = SxxSum + Sxx[0:specLen]

    # periodogram
    Sxx = (SxxSum/float(nFrames))

    return Sxx, freqS

def lpc2form(a, Fs=1.0):
    '''
    Convert all-pole coefficients to resonance frequencies
    and bandwidths

    a: LPC coefficients (all-pole coefficients excluding order 0)
    Fs: sampling rate
    '''
    RTS = np.roots(np.concatenate(([1],a)));

    # roots are complex conjugate pairs
    RTS = RTS[np.imag(RTS)>=0];
    AngZ = np.arctan2(np.imag(RTS),np.real(RTS));

    # Convert normalised frequency to freq.
    nFreq = AngZ*(Fs/(2*np.pi))
    Indices = np.argsort(nFreq);
    FreqS = nFreq[Indices]
    FreqS = FreqS[FreqS>0]

    # Bandwidths are the distance to the unit circle
    BW = -1/2*(Fs/(2*np.pi))*np.log(np.abs(RTS[Indices]))

    return FreqS, BW

def lpc2form_full(a, Fs=1.0, npts=1024):
    FreqS, BW = lpc2form(a, Fs)
    omega, h = sig.freqz([1],np.concatenate(([1], a)), worN=npts)
    f = omega/np.pi * Fs/2
    pks = PeakFinder(x=f, y=np.abs(h))
    pks.refine_all()

    return FreqS, BW, pks.pos, pks.val

def Formants(w, Fs, tWind=0.025, tHop=0.0125,
                    fMin=50, fMax=5500, bwMax=400,
                    modelOrd=10, hpFreq=50, full=False):
    '''Estimate formants from waveform w with sample rate Fs

       tWind:      window length in seconds
       tHop:       hop length in seconds
       fMin:       minimum frequency of formant in Hz
       fMax:       maximum frequency of formant in Hz
                    (determines resampling rate)
       bwMax:      maximum bandwidth (Hz)
       modelOrder: model order for linear prediction (LPC)
       hpFreq:     cutoff frequency of pre-emphasis filter
                    (high-pass, 1st order)
       full:       also calclate amplitudes and freqs of peaks
    '''

    # pre-emphasise
    #
    if hpFreq>0:
        a=np.exp(-2.*np.pi*hpFreq/float(Fs));
        #preEmphA = [a,1-a];
        #wo = sig.lfilter([1],preEmphA,w);
        wo=w
        wo[:-1] -= wo[1:]
    else:
        wo=w

    # resample the original wave file
    # AnalysisFs = 8000;

    underSample = int(Fs/fMax/2);
    FsO = Fs;

    # Fourier method: can be slow!
    #w = sig.resample(wo,len(wo)/underSample);

    # Resample: polyhase method (only in scipy v18.1)
    w = sig.resample_poly(wo,1,underSample);

    Fs = int(FsO*len(w)/float(len(wo)));
    Fsf = float(Fs)

    wLen = len(w);

    hopLenSam = int(round(Fs*tHop));
    windowLenSam = int(round(Fs*tWind));
    #print 'Formant:         Fs={}; wLen={}; hop={}'.format(Fs,windowLenSam,hopLenSam)

    dt = 1./Fs;
    nFrames = int(np.floor((wLen-windowLenSam-1)/hopLenSam))

    Form = np.nan*np.ones((nFrames,int(modelOrd/2)));
    BandWidths = np.nan*np.ones((nFrames,int(modelOrd/2)));
    if full:
        Peaks = np.nan*np.ones((nFrames,int(modelOrd/2)));
        Amplitudes = np.nan*np.ones((nFrames,int(modelOrd/2)));
    Time = np.arange(nFrames+0)*hopLenSam/Fsf+windowLenSam/Fsf/2


    for FN in np.arange(nFrames):
        I0 = (FN)*hopLenSam;
        Iend = I0 + windowLenSam;
        X = w[I0:Iend];

        XW = X*sig.hamming(len(X));
        #XW = X*sig.gaussian(len(X),0.4);

        # pre-emphasis filter
        # all-pole high pass filter

        #PreEmph = [1 0.63];
        #XW = filter(1,PreEmph,XW);

        # call LPC
        # A, err, rcoeff = lpc(XW,modelOrd);
        A = lpc(XW,modelOrd);

        if full:
            FreqS, BW, pkF, pkA = lpc2form_full(A, Fs)
        else:
            FreqS, BW = lpc2form(A, Fs)
        NN = 0
        for KK in range(len(FreqS)):
            if (FreqS[KK] > fMin and FreqS[KK] < fMax-fMin and BW[KK] <bwMax):
                Form[FN, NN] = FreqS[KK]
                BandWidths[FN, NN] = BW[KK]
                if full:
                    if len(pkF)>0:
                        idx = np.argmin(np.abs(pkF-FreqS[KK]))
                        Peaks[FN, NN] = pkF[idx]
                        Amplitudes[FN, NN] = pkA[idx]
                NN = NN + 1
            else:
                #print('Rejected f={}, bw={}'.format(FreqS[KK],BW[KK]))
                pass
    if full:
        return Time, Form, BandWidths, Peaks, Amplitudes
    else:
        return Time, Form, BandWidths

def rmsWind(w, nwind=256, nhop=None, windfunc=np.ones, sr=1):
    '''
    calculate RMS values in window chunks of data
    '''

    if not nhop:
        nhop=nwind/2

    i=0

    nw=len(w)

    tl=[]
    rl=[]

    wvr = windfunc(nwind)
    wvnorm = np.sqrt(sum(wvr**2)/nwind)

    wv = wvr/wvnorm

    while i<nw-nwind:
        rl.append(np.std(w[i:i+nwind]*wv))
        tl.append((i+nwind/2)/float(sr))
        i+=nhop

    return np.array(tl),np.array(rl)


def FricativeDataFromSnip(w, nwind=256, sr=1):
    fricdata = dict()

    freq,Sww = sig.welch(w,nperseg=nwind,fs=sr)

    logSww = 10*np.log10(Sww)

    # Peak position
    imax = np.argmax(Sww)
    maxref,maxv = refine_max(logSww,imax)
    fmax = np.interp(maxref,np.arange(len(Sww)),freq)
    fricdata['f_max']=fmax

    # RMS
    rms = np.std(w)
    fricdata['rms']=rms

    # spectral slopes
    p_left = np.polyfit(freq[:imax+1],logSww[:imax+1],1)
    slp_left = p_left[0]*1000

    p_right = np.polyfit(freq[imax:],logSww[imax:],1)
    slp_right = p_right[0]*1000


    fricdata['slope_left']=(slp_left)
    fricdata['slope_right']=(slp_right)


    cent,stdev,skew,kurt,_=DistribMoments(freq,Sww)
    fricdata['cent']=(cent)
    fricdata['stdev']=(stdev)
    fricdata['skew']=(skew)
    fricdata['kurt']=(kurt)

    return fricdata

def FricativeDataFromWav(wavname, intervals=[], pos=[0.5],
                        hpFreq=50,wsize=1024):
    sr,w = wavfile.read(wavname)
    if hpFreq>0:
        wo = ftf.preemph(w,Fs=sr,hpFreq=hpFreq)
    else:
        wo = w

    tamp,amp = rmsWind(w,nwind=wsize*4)

    fdict = {pp:pd.DataFrame() for pp in pos}

    for st,end in intervals:
        kept=[]
        for pp in pos:
            print('{:.3f}-{:.3f} @ {}'.format(st,end,pp))
            try:
                duration = end-st
                imed = int((st+duration*(pp))*sr)
                imin = int(imed-wsize/2)
                imax = int(imed+wsize/2)
                if imin < st*sr:
                    imin = int(row[('all',start_col)]*sr)
                    imax = int(imin+wsize)
                if imax > end*sr:
                    imax = int(row[('all',end_col)]*sr)
                    imin = int(imax - wsize)

                ww = wo[imin:imax]

                fricdata = FricativeData(ww,sr=sr,nwind=wsize)

                fdict[pos].append(fricdata)
            except ValueError as e:
                print(' ERROR at {}-{}'.format(st,end))
                print(e)
                fdict[pos].append([])
    return fdict
