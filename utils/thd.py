import numpy as np

def calculate_thd(signal, sample_rate, fundamental_freq):
    """计算THD(总谐波失真)
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        fundamental_freq: 基频
        
    Returns:
        thd: THD值(百分比)
    """
    # 进行FFT
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # 找到基频幅值
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fundamental = np.abs(spectrum[fund_idx])
    
    # 找到谐波幅值
    harmonics = []
    for n in range(2, 11):  # 计算2-10次谐波
        harm_idx = np.argmin(np.abs(freqs - n*fundamental_freq))
        harmonics.append(np.abs(spectrum[harm_idx]))
    
    # 计算THD
    thd = np.sqrt(np.sum(np.array(harmonics)**2)) / fundamental * 100
    
    return thd

def calculate_thd_plus_n(signal, sample_rate, fundamental_freq):
    """计算THD+N(总谐波失真加噪声)
    
    Args:
        signal: 输入信号
        sample_rate: 采样率
        fundamental_freq: 基频
        
    Returns:
        thd_n: THD+N值(百分比)
    """
    # 进行FFT
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # 找到基频幅值和索引
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fundamental = np.abs(spectrum[fund_idx])
    
    # 计算总能量
    total_power = np.sum(np.abs(spectrum)**2)
    
    # 移除基频分量
    spectrum[fund_idx] = 0
    if fund_idx > 0:  # 处理共轭复数
        spectrum[-fund_idx] = 0
        
    # 计算剩余能量(谐波+噪声)
    distortion_and_noise = np.sqrt(np.sum(np.abs(spectrum)**2))
    
    # 计算THD+N
    thd_n = distortion_and_noise / fundamental * 100
    
    return thd_n

