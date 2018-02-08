from numpy import array, pad, array, ndarray
import wave
import struct

def everyOther (v, offset=0):
    return [v[i] for i in range(offset, len(v), 2)]

def wavInfo(fname):
    wav = wave.open (fname, "r")
    params = wav.getparams ()
    wav.close()
    return params
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)

def wavLoad (fname, startTime=0.0, endTime=None):
    wav = wave.open (fname, "r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()
    if startTime > 0.0:
        wav.setpos(int(startTime*float(framerate*nchannels))/nchannels)
    
    if endTime:
        nrdframes = int((endTime-startTime)*float(framerate*nchannels))/nchannels
    else:
        nrdframes = nframes-wav.tell()
        
    frames = wav.readframes (nrdframes * nchannels)
    out = struct.unpack_from ("%dh" % nrdframes * nchannels, frames)
    
    # Convert 2 channles to numpy arrays
    if nchannels == 2:
       left = array (list (everyOther (out, 0)))
       right = array (list  (everyOther (out, 1)))
       return framerate, array(left,right)
    else:
       left = array (out)
       #right = left
       return framerate, left

def wavCopy (infile, outfile, startTime=0.0, endTime=None):
    inwav = wave.open (infile, "r")
    outwav = wave.open (outfile, "w")
    (nchannels, sampwidth, framerate, 
        nframes, comptype, compname) = inwav.getparams ()

    if startTime > 0.0:
        inwav.setpos(int(startTime*float(framerate*nchannels))/nchannels)
    
    if endTime:
        nrdframes = int((endTime-startTime)*float(framerate*nchannels))/nchannels
    else:
        nrdframes = nframes-inwav.tell()
    
    outwav.setnchannels(nchannels)
    outwav.setsampwidth(sampwidth)
    outwav.setframerate(framerate)
    
    for ii in range(nrdframes):
        frames = inwav.readframes (nchannels)
        outwav.writeframes(frames)
    
    outwav.close()
    inwav.close()

def wavSave (data, framerate, fname, sampwidth=2):
    wav = wave.open (fname, "w")
    wav.setframerate(framerate)
    wav.setsampwidth(sampwidth)
    if hasattr(data[0], '__len__'):
        nchan = len(data[0])
        values = [struct.pack('h',int(d)) for dd in data for d in dd ]
    else:
        nchan = 1
        values = [struct.pack('h',int(d)) for d in data]
    wav.setnchannels(nchan)
        
    valstr = ''.join(values)
    wav.writeframes (valstr)
    
    wav.close

    
def play(w,sr):
    if type(w) is not ndarray:
        w=array(w)
    try:
        nchan = w.shape[1]
    except IndexError:
        nchan=1
        if w.shape[0] <20000:
            w=pad(w,pad_width=(4000,4000),mode='constant', constant_values=(0,0))
    
    w16 = w.astype('int16').tobytes()
    # Open stream with correct settings
    
    import pyaudio
    pya = pyaudio.PyAudio()
    stream = pya.open(format=pya.get_format_from_width(width=2), channels=nchan, rate=sr, output=True)
    # Assuming you have a numpy array called samples
    stream.write(w16)
    stream.stop_stream()
    stream.close()

    pya.terminate()

    
