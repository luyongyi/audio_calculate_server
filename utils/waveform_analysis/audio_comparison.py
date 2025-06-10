import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from utils.waveform_analysis._common import load
import soundfile as sf
import os

# 参数设置
class AudioCompareParams:
    """音频比较参数设置"""
    # 高通滤波器参数
    HIGH_PASS_CUTOFF = 200  # 高通滤波器截止频率（Hz）
    FILTER_ORDER = 5       # 滤波器阶数
    
    # 滑窗参数
    WINDOW_SIZE = 1024     # 窗大小（采样点数）
    OVERLAP = 0.5         # 重叠比例（0-1之间）
    HOP_SIZE = None       # 跳跃大小，根据WINDOW_SIZE和OVERLAP自动计算
    
    # 阈值参数
    MAX_DIFF_DB_THRESHOLD = 3.0  # 最大差值阈值（dB）
    
    @classmethod
    def get_hop_size(cls):
        """计算跳跃大小"""
        if cls.HOP_SIZE is None:
            cls.HOP_SIZE = int(cls.WINDOW_SIZE * (1 - cls.OVERLAP))
        return cls.HOP_SIZE

def linear_to_db(linear_value):
    """将线性值转换为分贝值
    
    Args:
        linear_value: 线性值
        
    Returns:
        float: 分贝值
    """
    return 20 * np.log10(linear_value + 1e-10)  # 添加小值避免log(0)

def butter_highpass(cutoff, fs, order=5):
    """设计高通滤波器
    
    Args:
        cutoff: 截止频率
        fs: 采样率
        order: 滤波器阶数
        
    Returns:
        b, a: 滤波器系数
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    """应用高通滤波器
    
    Args:
        data: 输入信号
        cutoff: 截止频率
        fs: 采样率
        order: 滤波器阶数
        
    Returns:
        filtered_data: 滤波后的信号
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def calculate_windowed_rms(signal, window_size, hop_size):
    """使用滑窗计算RMS值
    
    Args:
        signal: 输入信号
        window_size: 窗大小
        hop_size: 跳跃大小
        
    Returns:
        rms_values: RMS值数组
        time_points: 对应的时间点
    """
    # 确保信号长度是窗大小的整数倍
    n_frames = 1 + (len(signal) - window_size) // hop_size
    rms_values = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size
        window = signal[start:end]
        rms_values[i] = np.sqrt(np.mean(window**2))
    
    return rms_values

def calculate_rms_difference(ref_file, deg_file):
    """计算两个音频文件的RMS差值
    
    Args:
        ref_file: 参考音频文件路径
        deg_file: 待比较音频文件路径
        
    Returns:
        dict: 包含RMS差值信息的字典
    """
    # 加载音频文件
    ref_data = load(ref_file)
    deg_data = load(deg_file)
    
    # 确保采样率相同
    if ref_data['fs'] != deg_data['fs']:
        raise ValueError("采样率不匹配")
    
    # 获取信号数据
    ref_signal = ref_data['signal']
    deg_signal = deg_data['signal']
    
    # 获取滑窗参数
    window_size = AudioCompareParams.WINDOW_SIZE
    hop_size = AudioCompareParams.get_hop_size()
    
    result = {
        'fs': ref_data['fs'],
        'channels': {
            'ref': ref_data['channels'],
            'deg': deg_data['channels']
        },
        'window_params': {
            'window_size': window_size,
            'hop_size': hop_size,
            'overlap': AudioCompareParams.OVERLAP
        }
    }
    
    # 处理单声道和双声道
    if ref_data['channels'] == 1 and deg_data['channels'] == 1:
        # 单声道比较
        ref_filtered = highpass_filter(ref_signal, 
                                     AudioCompareParams.HIGH_PASS_CUTOFF, 
                                     ref_data['fs'],
                                     AudioCompareParams.FILTER_ORDER)
        deg_filtered = highpass_filter(deg_signal, 
                                     AudioCompareParams.HIGH_PASS_CUTOFF, 
                                     deg_data['fs'],
                                     AudioCompareParams.FILTER_ORDER)
        
        # 计算滑窗RMS
        ref_rms = calculate_windowed_rms(ref_filtered, window_size, hop_size)
        deg_rms = calculate_windowed_rms(deg_filtered, window_size, hop_size)
        
        # 计算RMS差值
        rms_diff = np.abs(ref_rms - deg_rms)
        
        # 计算dB值
        max_diff_db = linear_to_db(np.max(rms_diff))
        
        result['mono'] = {
            'mean_diff': float(np.mean(rms_diff)),
            'max_diff': float(np.max(rms_diff)),
            'max_diff_db': float(max_diff_db),
            'std_diff': float(np.std(rms_diff)),
            'rms_diff': rms_diff,
            'time_points': np.arange(len(rms_diff)) * hop_size / ref_data['fs']
        }
        
    elif ref_data['channels'] == 2 and deg_data['channels'] == 2:
        # 双声道分别比较
        # 左声道
        ref_left = highpass_filter(ref_signal[:, 0], 
                                 AudioCompareParams.HIGH_PASS_CUTOFF, 
                                 ref_data['fs'],
                                 AudioCompareParams.FILTER_ORDER)
        deg_left = highpass_filter(deg_signal[:, 0], 
                                 AudioCompareParams.HIGH_PASS_CUTOFF, 
                                 deg_data['fs'],
                                 AudioCompareParams.FILTER_ORDER)
        
        # 右声道
        ref_right = highpass_filter(ref_signal[:, 1], 
                                  AudioCompareParams.HIGH_PASS_CUTOFF, 
                                  ref_data['fs'],
                                  AudioCompareParams.FILTER_ORDER)
        deg_right = highpass_filter(deg_signal[:, 1], 
                                  AudioCompareParams.HIGH_PASS_CUTOFF, 
                                  deg_data['fs'],
                                  AudioCompareParams.FILTER_ORDER)
        
        # 计算滑窗RMS
        ref_rms_left = calculate_windowed_rms(ref_left, window_size, hop_size)
        deg_rms_left = calculate_windowed_rms(deg_left, window_size, hop_size)
        ref_rms_right = calculate_windowed_rms(ref_right, window_size, hop_size)
        deg_rms_right = calculate_windowed_rms(deg_right, window_size, hop_size)
        
        # 计算RMS差值
        rms_diff_left = np.abs(ref_rms_left - deg_rms_left)
        rms_diff_right = np.abs(ref_rms_right - deg_rms_right)
        
        # 计算dB值
        max_diff_left_db = linear_to_db(np.max(rms_diff_left))
        max_diff_right_db = linear_to_db(np.max(rms_diff_right))
        
        time_points = np.arange(len(rms_diff_left)) * hop_size / ref_data['fs']
        
        result['stereo'] = {
            'left': {
                'mean_diff': float(np.mean(rms_diff_left)),
                'max_diff': float(np.max(rms_diff_left)),
                'max_diff_db': float(max_diff_left_db),
                'std_diff': float(np.std(rms_diff_left)),
                'rms_diff': rms_diff_left,
                'time_points': time_points
            },
            'right': {
                'mean_diff': float(np.mean(rms_diff_right)),
                'max_diff': float(np.max(rms_diff_right)),
                'max_diff_db': float(max_diff_right_db),
                'std_diff': float(np.std(rms_diff_right)),
                'rms_diff': rms_diff_right,
                'time_points': time_points
            }
        }
    else:
        raise ValueError("声道数不匹配：参考文件和待比较文件的声道数必须相同")
    
    return result

def plot_rms_difference(rms_diff, fs, save_path=None, channel_type='mono'):
    """绘制RMS差值图
    
    Args:
        rms_diff: RMS差值数组或字典（对于立体声）
        fs: 采样率
        save_path: 保存图片的路径
        channel_type: 声道类型，'mono'或'stereo'
        
    Returns:
        str: 保存的图片路径
    """
    plt.figure(figsize=(12, 6))
    
    if channel_type == 'mono':
        plt.plot(rms_diff['mono']['time_points'], 
                rms_diff['mono']['rms_diff'], 
                label='单声道')
    else:
        plt.plot(rms_diff['stereo']['left']['time_points'], 
                rms_diff['stereo']['left']['rms_diff'], 
                label='左声道')
        plt.plot(rms_diff['stereo']['right']['time_points'], 
                rms_diff['stereo']['right']['rms_diff'], 
                label='右声道')
        plt.legend()
    
    plt.xlabel('时间 (秒)')
    plt.ylabel('RMS差值')
    plt.title('音频RMS差值分析')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        plt.show()
        return None 