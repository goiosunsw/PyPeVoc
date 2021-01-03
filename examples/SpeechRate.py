#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  SpeachRate.py
#  
#  estimate the speech rate of a file, 
#  also generating segmentation textgrids
#
#  Copyright 2017 Andre Almeida <goios@goios-UX305UA>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import SpeechSegmenter as ss
import SpeechChunker as sc
from scipy.io import wavfile
import numpy as np
import sys
import os

def segment_recording(sound_files):
    w=[]
    for ff in sound_files:
        sr,wi=wavfile.read(ff)
        w.append(wi)
    
    w=np.vstack(w).T
    sys.stderr.write("Read {} files, {} channels, {} samples\n"\
                     .format(len(sound_files),w.shape[1],w.shape[0]))
    sys.stderr.write("Segmenting audio\n")
    seg=sc.MultiChannelSegmenter(w,sr=sr)
    sys.stderr.write("Found {} chunks\n".format(len(seg.label)))
    return (seg.tst,seg.tend,seg.label)
    
def analyse_rec(sound_files, output_dir='.'):
    # segment recordings
    w=[]
    for ff in sound_files:
        sr,wi=wavfile.read(ff)
        w.append(wi)
    
    w=np.vstack(w).T
    sys.stderr.write("Read {} files, {} channels, {} samples\n"\
                     .format(len(sound_files),w.shape[1],w.shape[0]))
    sys.stderr.write("Segmenting audio\n")
    if w.shape[1]>1:
        seg=sc.MultiChannelSegmenter(w,sr=sr,min_len=args.min_silence)
    else:
        #w=w.squeeze()
        seg = sc.SilenceDetector(w.squeeze(), sr=sr, method = 'pct01',
                                min_len=args.min_silence)
        seg.label = [1 for tst in seg.tst]
        seg.centers = np.array([[0,0],[1,0]])
                                
        
    seg.to_textgrid(os.path.join(output_dir,"sources.TextGrid"))
    sys.stderr.write("Found {} chunks\n".format(len(seg.label)))
    
    intervals = (seg.tst,seg.tend,seg.label)
    
    # segment syllables for each channel
    for lab in set(seg.label):
        vi = [(ii[0],ii[1]) for ii in zip(*intervals) if ii[2]==lab]
        source = int(lab)
        # find the best channel to segment source
        chan = np.argmax(seg.centers[lab,:])
        
        syl=ss.SyllableSegmenter(w[:,chan],sr=sr,voice_intervals=vi)
        syl.segment_amplitude_bumps()
        syl.classify_voicing()
        syl.to_textgrid(os.path.join(output_dir,'voiced_syllables_{}.TextGrid'.format(lab)))
        # output spreadsheet
        df = syl.to_pandas()
        df.to_excel(os.path.join(output_dir,'syllables_{}.xls'.format(lab)))


def process_file_list(batch_file):
    import logging
    file_seq=[]
    with open(batch_file) as f:
        for line in f:
            files = [it.strip() for it in line.split(',') if len(it.strip())>0]
            if len(files)>0:
                basedir, filename = os.path.split(files[0])
                try:
                    analyse_rec(files, output_dir=basedir)
                except Exception as e:
                    message = 'ERROR while processing files:\n'
                    for f in files:
                        message+=f
                    message+='/n'
                    logging.exception(message)
                    #~ sys.stderr.write('ERROR while processing files:\n')
                    #~ for f in files:
                        #~ sys.stderr.write(f+'\n')
                    #~ sys.stderr.write(str(e))
                    #~ sys.stderr.write('\n')
                    #~ sys.stderr.write(e.__doc__ )
                    #~ sys.stderr.write('\n')
    return 0
    
def main(args):

    sound_files = args.infiles
    print(sound_files)
    
    if args.batch:
        process_file_list(args.batch)
        
    else:
        if sound_files:  
            analyse_rec(sound_files)
        else:
            sys.stderr.write('Input files or batch list (-b) are required!\n')
        

    return 0

if __name__ == '__main__':
    import sys
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", nargs='?', default = '',
        help = "output file name")
    ap.add_argument("-n", "--min-silence", nargs='?', default = '0.3', type=float,
        help = "minimum silence duration in seconds")
    ap.add_argument("-b", "--batch", nargs='?', 
        help = "input file list for batch processing")

    ap.add_argument("-s", "--start", type=float, nargs='?', default = '0',
        help = "start time")
    ap.add_argument("-e", "--end", type=float, nargs='?', default = '-1',
        help = "end time")

    
    ap.add_argument('infiles', nargs='*', help='Input sound files (required if not batch)')
    
    args = ap.parse_args()

    

    sys.exit(main(args))
