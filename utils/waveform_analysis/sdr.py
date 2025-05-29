import numpy as np

def compute_SDR(pred, gt, eps=1e-7):
    """计算SDR，支持单声道或立体声输入
    
    Args:
        pred: 输入信号，可以是单声道(n_samples,)或立体声(n_samples, n_channels)
        gt: 参考信号，可以是单声道(n_samples,)或立体声(n_samples, n_channels)
        eps: 防止除零的小量，默认为1e-7
        
    Returns:
        float: SDR值（dB）
    """
    # 如果是立体声，转换为单声道（取平均值）
    if len(pred.shape) > 1:
        pred = np.mean(pred, axis=1)
    if len(gt.shape) > 1:
        gt = np.mean(gt, axis=1)
    
    # 保证形状一致
    min_len = min(len(pred), len(gt))
    pred, gt = pred[:min_len], gt[:min_len]
    
    num = np.sum(gt**2) + eps
    den = np.sum((gt - pred)**2) + eps
    return 10 * np.log10(num / den) 