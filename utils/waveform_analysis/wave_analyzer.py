#!/usr/bin/env python

import numpy as np
from numpy import log10, pi, convolve, mean
from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter
from scipy.signal import butter
from scipy.signal import sosfilt
#from scikits.audiolab import Sndfile, Format

def A_weighting(Fs):
    """Design of an A-weighting filter.
    
    [B,A] = A_weighting(Fs) designs a digital A-weighting filter for 
    sampling frequency Fs. Usage: Y = FILTER(B,A,X). 
    Warning: Fs should normally be higher than 20 kHz. For example, 
    Fs = 48000 yields a class 1-compliant filter.
    
    Originally a MATLAB script. Also included ASPEC, CDSGN, CSPEC.
    
    Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
            couvreur@thor.fpms.ac.be
    Last modification: Aug. 20, 1997, 10:00am.
    
    http://www.mathworks.com/matlabcentral/fileexchange/69
    http://replaygain.hydrogenaudio.org/mfiles/adsgn.m
    Translated from adsgn.m to PyLab 2009-07-14 endolith@gmail.com
    
    References: 
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    
    NUMs = [(2 * pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = convolve([1, +4 * pi * f4, (2 * pi * f4)**2], 
                    [1, +4 * pi * f1, (2 * pi * f1)**2], mode='full')
    DENs = convolve(convolve(DENs, [1, 2 * pi * f3], mode='full'), 
                                   [1, 2 * pi * f2], mode='full')
    
    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, Fs)

def A_weight(signal, samplerate):
    """Return the given signal after passing through an A-weighting filter"""
    B, A = A_weighting(samplerate)
    return lfilter(B, A, signal)

# From matplotlib.mlab
def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a)**2))

def ac_rms(signal):
    """Return the RMS level of the signal after removing any fixed DC offset"""
    return rms_flat(signal - mean(signal))

def dB(level):
     """Return a level in decibels.
     
     Decibels are relative to the RMS level of a full-scale square wave 
     of peak amplitude 1.0 (dBFS).
     
     A full-scale square wave is 0 dB
     A full-scale sine wave is -3.01 dB

     """
     return 20 * log10(level+0.0000001)
# def display(header, results):
#     """Display header string and list of result lines"""
#     try:
#         import easygui
#     except ImportError:
#         #Print to console
#         print 'EasyGUI not installed - printing output to console\n'
#         print header
#         print '-----------------'
#         print '\n'.join(results)
#     else:
#         # Pop the stuff up in a text box
#         title = 'Waveform properties'
#         easygui.textbox(header, title, '\n'.join(results))

# def histogram(signal):
#     """Plot a histogram of the sample values"""
#     try:
#         from matplotlib.pyplot import hist, show
#     except ImportError:
#         print 'Matplotlib not installed - skipping histogram'
#     else:
#         print 'Plotting histogram'
#         hist(signal) #parameters
#         show()

def properties(signal, samplerate):
    """Return a list of some wave properties for a given 1-D signal"""
    signal_level = ac_rms(signal)
    peak_level = max(max(signal.flat),-min(signal.flat))
    crest_factor = peak_level/signal_level
    
    # Apply the A-weighting filter to the signal
    weighted = A_weight(signal, samplerate)
    weighted_level = ac_rms(weighted)
    
    return [
    'DC offset: %f (%.3f%%)' % (mean(signal), mean(signal)*100),
    'Crest factor: %.3f (%.3f dB)' % (crest_factor, dB(crest_factor)),
    'Peak level: %.3f (%.3f dB)' % (peak_level, dB(peak_level)), # Does not take intersample peaks into account!
    'RMS level: %.3f (%.3f dB)' % (signal_level, dB(signal_level)),
    'A-weighted: %.3f (%.3f dB)' % (weighted_level, dB(weighted_level)),
    'A-difference: %.3f dB' % dB(weighted_level/signal_level),
    '-----------------',
    ]
    
# def analyze(filename):
#     wave_file = Sndfile(filename, 'r')
#     signal = wave_file.read_frames(wave_file.nframes)
    
#     header = 'dB values are relative to a full-scale square wave'
    
#     results = [
#     'Properties for "' + filename + '"',
#     str(wave_file.format),
#     'Channels: %d' % wave_file.channels,
#     'Sampling rate: %d Hz' % wave_file.samplerate,
#     'Frames: %d' % wave_file.nframes,
#     'Length: ' + str(wave_file.nframes/wave_file.samplerate) + ' seconds',
#     '-----------------',
#     ]
    
#     # Currently only handles mono or stereo files
#     # If both channels are identical, it will only show one properties sheet
    
#     if wave_file.channels == 1:
#         # Monaural
#         results += properties(signal, wave_file.samplerate)
#     elif wave_file.channels == 2:
#         # Stereo
#         if np.array_equal(signal[:,0],signal[:,1]):
#             results += ['Left and right channels are identical:']
#             results += properties(signal[:,0], wave_file.samplerate)
#         else:
#             results += ['Left channel:']
#             results += properties(signal[:,0], wave_file.samplerate)
#             results += ['Right channel:']
#             results += properties(signal[:,1], wave_file.samplerate)
#     else:
#         # Multi-channel
#         for ch_no, channel in enumerate(signal.transpose()):
#             results += ['Channel %d:' % (ch_no + 1)]
#             results += properties(channel, wave_file.samplerate)
    
#     display(header, results)
    
#     plot_histogram = False
#     if plot_histogram:
#         histogram(signal)

def find_peak(signal):
    """找到信号的峰值点
    
    Args:
        signal: 输入信号
        
    Returns:
        peak_idx: 峰值点的索引
    """
    return np.argmax(np.abs(signal))

def calculate_energy(signal, start_idx, end_idx):
    """计算信号在指定范围内的能量
    
    Args:
        signal: 输入信号
        start_idx: 起始索引
        end_idx: 结束索引
        
    Returns:
        energy: 能量值
    """
    return np.sum(signal[start_idx:end_idx]**2)

def calculate_C80(signal, sample_rate):
    """计算C80值
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        
    Returns:
        C80: C80值(dB)
    """
    # 找到峰值点
    peak_idx = find_peak(signal)
    
    # 计算前80ms的能量
    ms_to_samples = int(sample_rate * 0.001)  # 1ms对应的采样点数
    early_start = max(0, peak_idx - 20 * ms_to_samples)
    early_end = peak_idx + 60 * ms_to_samples
    early_energy = calculate_energy(signal, early_start, early_end)
    
    # 计算后80ms的能量
    late_start = peak_idx + 60 * ms_to_samples
    late_end = peak_idx + 140 * ms_to_samples
    late_energy = calculate_energy(signal, late_start, late_end)
    
    # 计算C80
    C80 = 10 * np.log10(early_energy / late_energy)
    
    return C80
def db_to_linear(db_value):
    """
    将分贝值转换为线性值
    :param db_value: 分贝值
    :return: 线性值
    """
    return round(1000000000 * 0.00002 * (10 ** (db_value / 20))) / 1000000000

def db_to_linear_correct(db_value):
    """
    将分贝值转换为线性值（标准实现）
    :param db_value: 分贝值
    :return: 线性值
    """
    return 10 ** (db_value / 20)

def get_octave_bands():
    """获取1/3倍频程的中心频率
    
    Returns:
        list: 中心频率列表(Hz)
    """
    # 标准1/3倍频程中心频率(Hz)
    return [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
            500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
            6300, 8000, 10000, 12500, 16000, 20000]

def calculate_octave_3(data, samplerate):
    """计算信号的1/3倍频程分析
    
    Args:
        data: 输入信号
        samplerate: 采样率
        
    Returns:
        dict: 包含每个频段RMS值的字典，键为中心频率
    """
    # 获取中心频率列表
    center_freqs = get_octave_bands()
    results = {}
    
    # 1/3 倍频程时，上下限系数 factor = 2^(±1/6)
    factor = 2 ** (1/6)
    nyquist = samplerate / 2.0
    
    for fc in center_freqs:
        # 计算带通滤波器的上下截止频率
        low = (fc / factor) / nyquist
        high = (fc * factor) / nyquist
        
        # 在可Nyquist范围内，且低频限不能小于0，高频限不能大于1
        low = max(low, 1e-6)     # 避免低于0
        high = min(high, 0.999) # 避免超过Nyquist
        
        # 确保low < high
        if low >= high:
            continue
        
        # 使用巴特沃斯带通滤波器
        sos = butter(4, [low, high], btype='band', output='sos')
        
        # 应用滤波器
        filtered = sosfilt(sos, data)
        
        # 计算RMS值
        rms_value = rms_flat(filtered)
        
        # 存储结果
        results[fc] = rms_value
    
    return results

if __name__ == '__main__':
    pass
    # import sys
    # if len(sys.argv) == 2:
    #     filename = sys.argv[1]
    #     analyze(filename)
    # else:
    #     print 'You need to provide a filename:\n'
    #     print 'python wave_analyzer.py filename.wav'
        