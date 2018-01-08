"""
Defines some useful functions for the estimation of transfer functions
"""

import numpy as np
import matplotlib.pyplot as pl
import scipy.signal as sig


def tfe_sig(y, x, *args, **kwargs):
    """estimate transfer function from x to y,
       see csd for calling convention"""
    fxy, sxy = sig.csd(y, x, *args, **kwargs)
    fxx, sxx = sig.csd(x, x, *args, **kwargs)
    return sxy / sxx, fxx


try:
    from matplotlib.mlab import psd, csd, cohere

    def tfe(y, x, *args, **kwargs):
        """estimate transfer function from x to y,
           see csd for calling convention"""
        sxy, fxy = csd(y, x, *args, **kwargs)
        sxx, fxx = psd(x, *args, **kwargs)
        return sxy / sxx, fxx


except ImportError:
    tfe = tfe_sig


def nextpow2(number):
    intlognum = int(np.log2(number))
    return 2**intlognum


def fft_filter(x, bands, gains):
    '''
    Filter signal x using FFT and IFFT
    * x input signal
    * bands: list of start and stop frequencies of each band
    * gains: start and stop gains in each band

    Example:

    y = FFTfilter(x, [(0,0.1),(0.1,1.0)], [(1.,1.),(0.,0.)])

    filters signal x low pass at 0.1 times the nyquist rate
      (sampling rate / 2)
    '''

    xf = np.fft.fft(x)
    nyq = len(xf)/2

    ffilter = np.zeros(len(xf))
    for bb, gg in zip(bands, gains):
        fmin = int(bb[0]*nyq)
        fmax = int(bb[1]*nyq)
        ffilter[fmin:fmax] = np.linspace(gg[0], gg[1], fmax-fmin)
        if fmin > 0:
            ffilter[-fmax+1:-fmin+1] = np.linspace(gg[1], gg[0], fmax-fmin)
        else:
            ffilter[-fmax+1:] = np.linspace(gg[1], gg[0], fmax-fmin-1)

    xf_filt = xf*ffilter
    return np.fft.ifft(xf_filt)


def smthderiv(ff, ph, rad=1):
    dph = []
    for i, phi in enumerate(ph):
        imin = max(0, i-rad)
        imax = min(len(ph), i+rad)
        pp = np.polyfit(ff[imin:imax], ph[imin:imax], 1)
        dph.append(pp[0])
    return np.array(dph)


def determineDelay(source, target, maxdel=2**16, ax=None):
    '''
    Determine the delay between two signals
    (based on correlation extrema)

    Parameters:
    * Signals
      - source
      - target
    * maxdel: maximum delay to look for (in both directions)
    '''
    sample_start = 0
    xd = source[sample_start:sample_start+maxdel]
    yd = target[sample_start:sample_start+maxdel]
    Cxx = np.correlate(xd, xd, 'full')
    Cxy = np.correlate(yd, xd, 'full')
    Pkx = np.argmax(np.abs(Cxx))
    Pky = np.argmax(np.abs(Cxy))
    if ax:
        try:
            ax.plot(Cxx)
        except AttributeError:
            fig, ax = pl.subplots(1)
            ax.plot(Cxx)
        ax.plot(Cxy)
        ax.axvline(Pkx, color='red')
        ax.plot(Pky, Cxy[Pky], 'o')

    delay = Pky-Pkx
    return delay


def transferogram(source, target, rate=1, start_time=0., delta_time=1.,
                  sample_duration=.5, window_duration=.125, window_hop=None):
    '''
    tfe, freqs, times, coherence = transferogram(...)

    Calculates a time-varying transfer function from source (x)
    to target (y) at intervals delta_time.

    Parameters:
    * source: source signal (reuqired)
    * target: target signal (required)
    * rate:   sampling rate
    * start_time: starting time for tfe calculations
    * delta_time: distance between calculations
    * sample_duration: length of signals used in tfe estimates
                       (longer than window_duration, used in averaging)
    * window_duration: inidvidual window length in tfe estimates
    * window_hop: hop between windows (defaults to window_duration/2)

    Returns:
    * tfe:   transfer functions (complex matrix NxM)
    * freqs: frequencies corresponding to tfe estimates (array size N)
    * times: times corresponding to tfe estimates (array size M)
    * coherence: coherence matrix MxN
    '''

    # convert time to samples
    sample_start = int(start_time*rate)
    sample_delta = int(delta_time*rate)
    sample_len = int(sample_duration*rate)

    if target is None:
        n_target = len(source)
    else:
        n_target = len(target)

    n_samples = min(len(source), n_target)
    sample_end = n_samples - sample_start - sample_len

    # windowing parameters
    nsamp_window = nextpow2(window_duration*rate)
    if window_hop:
        nsamp_window_hop = nextpow2(window_hop*rate)
    else:
        nsamp_window_hop = nsamp_window/2

    noverlap = nsamp_window - nsamp_window_hop

    resp = []
    coherence = []
    times = []

    if target is None:
        for ii in np.arange(sample_start, sample_end, sample_delta):
            block_resp, freq = psd(source[ii:ii+sample_len],
                                   NFFT=nsamp_window,
                                   noverlap=noverlap, Fs=rate)
            block_coh = []
            times.append((ii+sample_len/2)/float(rate))
            resp.append(block_resp)
            coherence.append(block_coh)
    else:
        for ii in np.arange(sample_start, sample_end, sample_delta):
            block_resp, freq = tfe(target[ii:ii+sample_len],
                                   source[ii:ii+sample_len], NFFT=nsamp_window,
                                   noverlap=noverlap, Fs=rate)
            block_coh, _ = cohere(target[ii:ii+sample_len],
                                  source[ii:ii+sample_len], NFFT=nsamp_window,
                                  noverlap=noverlap, Fs=rate)
            times.append((ii+sample_len/2)/float(rate))
            resp.append(block_resp)
            coherence.append(block_coh)

    return np.array(resp).T, freq, np.array(times), np.array(coherence).T


def block_delay(source, target, window=None):
    if window is None:
        window = np.ones(len(source))
    wind_source = window*source
    wind_target = window*target

    corr_st = np.correlate(wind_source, wind_target, "full")

    return np.argmax(corr_st)-len(source), np.max(corr_st)


def maxdelwind(source, target, rate=1, start_time=0., delta_time=1.,
               sample_duration=.5):
    '''
    delay, times = maxdelwid(...)

    Calculates a time-varying delay function from source (x)
    to target (y) at intervals delta_time.

    Parameters:
    * source: source signal (reuqired)
    * target: target signal (required)
    * rate:   sampling rate
    * start_time: starting time for tfe calculations
    * delta_time: distance between calculations
    * sample_duration: length of signals used in tfe estimates
                       (longer than window_duration, used in averaging)
    Returns:
    * delay: max delay array
    * times: times corresponding to delay estimates (array size M)
    '''
    
    # convert time to samples
    sample_start = int(start_time*rate)
    sample_delta = int(delta_time*rate)
    sample_len = int(sample_duration*rate)

    window = np.ones(sample_len)

    n_samples = min(len(source), len(target))
    sample_end = n_samples - sample_start - sample_len

    delay = []
    corr_strength = []
    times = []

    for block_start in np.arange(sample_start, sample_end, sample_delta):
        block_end = block_start + sample_len
        target_block = sig.detrend(target[block_start:block_end])
        source_block = sig.detrend(source[block_start:block_end])
        block_del, block_corr = block_delay(target_block, source_block,
                                            window=window)
        times.append((block_start+sample_len/2)/float(rate))
        delay.append(block_del/float(rate))
        corr_strength.append(block_corr)

    return np.array(delay), np.array(corr_strength), np.array(times)


def plot_time_freq(tf_matrix, freq=None, time=None, ax=None, mask=None):
    if time is None:
        time = np.arange(tf_matrix.shape[1])

    if freq is None:
        freq = np.arange(tf_matrix.shape[0])

    if ax is None:
        fig, ax = pl.subplots(1)

    if mask is not None:
        tf_matrix[np.logical_not(mask)] = np.nan

    ax.imshow(tf_matrix, aspect='auto', origin='lower',
              extent=[min(time), max(time), min(freq), max(freq)])

