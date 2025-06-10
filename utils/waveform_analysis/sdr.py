import torch
import numpy as np

def compute_SDR(pred, gt, eps=1e-7):
    """计算SDR，输入均为1D torch.Tensor或(n_samples,) ndarray"""
    if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):   gt = torch.from_numpy(gt)
    # 保证形状一致
    min_len = min(pred.shape[-1], gt.shape[-1])
    pred, gt = pred[..., :min_len], gt[..., :min_len]
    num = torch.sum(gt**2) + eps
    den = torch.sum((gt - pred)**2) + eps
    return 10 * torch.log10(num / den) 