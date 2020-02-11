import argparse
import pandas
import matplotlib.pyplot as plt
from scipy.io import wavfile

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('soundfile', help='Sound file')
    parser.add_argument('descfile', help='Descriptor csv file')
    return parser.parse_args() 

if __name__ == '__main__':
    args = parse_args()
    sr, w = wavfile.read(args.soundfile)
    #df=pandas.read_csv(args.descfile,names=['t_start','t_end','label','f0','RMS','Harm','F1','F2','F3','F4','F5'],
    #                   index_col=False)
    df =pandas.read_csv(args.descfile,index_col=0) 
    
    fig,ax = plt.subplots(3,sharex=True,figsize=(6,8))
    ax[0].specgram(w,Fs=sr,NFFT=1024)
    for ir, row in df.iterrows():
        ts = row['t_start']
        te = row['t_end']
        if row['label'].find('START')>-1:
            color='k'
        else:
            color='r'
        if row['label'].find('END')>-1:
            for axi in ax:
                axi.axvline(te, color='k',alpha=.5)
        for axi in ax:
            axi.axvline(ts, color=color, alpha=.5)
        
    tm = (df['t_start']+df['t_end'])/2
    ax[0].plot(tm,df['f0'],'o-',color='k')
    ax[0].plot(tm,df['Centroid'],'o-',color='blue')
    for ii in range(1,5):
        ax[0].plot(tm,df['F%d'%ii],'o-',color='r')
    ax[1].semilogy(tm,df['RMS'])
    ax[2].plot(tm,df['Harmonicity'])
    

    plt.show()
    