from scipy import signal,fft
import numpy as np
import scipy

def spectrum(data,fftSize,rate=48000,overlap=0.5):
    '''
    与audition对应频率分析图表
    ：param data：数据段
    ：param fftSize：分析fft大小，一般为512/1024/2048/4096/...等
    ：param rate：采样率，用于横坐标对齐
    ：param overlap：重叠率,一般为50%，可以自行选择
    '''
    dataSplit=[]
    window=signal.windows.hann(fftSize)
    splitLinespace=range(0,len(data),int(fftSize*overlap)) #会漏掉最后一部分数据，但是不多，问题不大
    print(splitLinespace)
    for i in splitLinespace:
        if(i+fftSize)>len(data):
            break
        dataSplit.append(data[i:i+fftSize])
    fftSum=np.zeros(fftSize//2)
    for i in dataSplit:                                     #分区
        i=i*window                                          #加窗
        fftData=abs(fft.fft(i))                             #fft
        fftSum+=np.array(fftData[:fftSize//2])              #求和
    #print(len(dataSplit))
    fftAvg=fftSum/len(dataSplit)                            #平均
    fftSpectrum=20*np.log10(abs(fftAvg)/(fftSize//4))       #幅度换算与校准
    fftFreq=np.linspace(0,rate//2,fftSize//2,endpoint=False)#横坐标计算
    return fftFreq,fftSpectrum  
def smooth(data, window_len=11, window='hanning'):
        if data.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        if data.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
        if window_len < 3:
            return data
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is not one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        
        s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        
        y = np.convolve(w / w.sum(), s, mode='valid')
        return y[int(window_len/2-1):-int(window_len/2)]
