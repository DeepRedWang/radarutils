
import numpy as np
from .phase import circle_fit
from scipy.signal import find_peaks

def vitalindex_filter(signal):
    """
    对输入的多维时间序列信号应用拟合，并计算Vital Index

    参数:
    ----------
    signal : np.ndarray
        输入的多维数组，形状为 (T, ...)，其中 T 是时间维度，其余维度表示空间或其他结构。

    返回:
    ----------
    map_VI : np.ndarray
        维度为 signal.shape[1:] 的数组，表示每个空间位置的活力指数（均值/标准差）。
        
    signal_circlefit : np.ndarray
        与输入信号形状相同的数组，表示每个位置的时间序列经过 circle_fit 拟合后的结果。
    """

    """
    Apply fitting to a multi-dimensional time series signal and compute the Vital Index (VI).

    Parameters:
    ----------
    signal : np.ndarray
        Input multi-dimensional array with shape (T, ...), where T is the time dimension,
        and the remaining dimensions represent spatial or other structures.

    Returns:
    ----------
    map_VI : np.ndarray
        An array with shape signal.shape[1:], representing the Vital Index (mean/std) at each spatial location.
        
    signal_circlefit : np.ndarray
        An array with the same shape as the input signal, containing the circle_fit output for each time series.
    """
    # 获取除了第一维之外的所有维度
    spatial_shape = signal.shape[1:]
    map_VI = np.zeros(spatial_shape)
    signal_circlefit = np.zeros_like(signal, dtype=np.complex64)

    # 遍历所有位置索引
    for idx in np.ndindex(spatial_shape):
        # 提取该位置处的向量 (shape: [N])
        vector = signal[(slice(None),) + idx]  # 相当于 DBF_data[:, i, j, ...]
        
        # 应用 circle_fit
        fitted, _ = circle_fit(vector)
        signal_circlefit[(slice(None),) + idx] = fitted
        amp = np.abs(fitted)
        map_VI[idx] = np.mean(amp) / (np.std(amp)+1e-8)
    return map_VI, signal_circlefit


def ACF_filter(signal):
    
    # 获取除了第一维之外的所有维度
    spatial_shape = signal.shape[1:]
    map_ACF = np.zeros(spatial_shape)
    signal_circlefit = np.zeros_like(signal, dtype=np.complex64)

    # 遍历所有位置索引
    for idx in np.ndindex(spatial_shape):
        # 提取该位置处的向量 (shape: [N])
        vector = signal[(slice(None),) + idx]  # 相当于 DBF_data[:, i, j, ...]
        
        # 应用 circle_fit
        fitted, _ = circle_fit(vector)
        signal_circlefit[(slice(None),) + idx] = fitted
        # amp = np.abs(fitted)
        # map_ACF[idx] = np.mean(amp) / (np.std(amp)+1e-8)

        phase = np.unwrap(np.angle(vector))
        autocorr = np.correlate(phase, phase, mode='full')
        new_autocorr = autocorr[-1*len(phase):]
        new_autocorr = new_autocorr / new_autocorr[0]
        peaks, _ = find_peaks(new_autocorr)
        if len(peaks) == 0:
            map_ACF[idx] = 0
        else:
            map_ACF[idx] = peaks[0]
        # lags = np.arange(-len(phase) + 1, len(phase))
        # center = len(phase) - 1  # lag=0 对应的位置




    return map_ACF, signal_circlefit



def loopback_filter(data, beta=0.97):
    """
    环回滤波器去背景
    :param data: ndarray [num_frames, num_range_bins] — 每帧一个快时间向量
    :param beta: 平滑因子
    :return: foreground: 去背景后的信号，同 shape
    """
    num_frames, num_bins = data.shape
    clutter = data[0]
    # clutter = np.mean(data[:20,:],axis=0)

    output = np.zeros_like(data)

    for k in range(num_frames):
        r_k = data[k]
        # 更新背景估计
        clutter = beta * clutter + (1 - beta) * r_k
        # 去背景
        output[k] = r_k - clutter

    return output


def moving_average(x, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode='same')  # 可选 'valid', 'full'

def differentiator_2order(signal, time_dim=0, pad_mode='zero'):
    """
    沿指定时间维进行高效平滑二阶差分，默认时间维为第 0 维。

    参数：
        signal: np.ndarray，任意形状的实/复信号
        time_dim: int，时间轴索引，默认 0 表示第一个维度
        pad_mode: str，边界处理方式：
            'zero'  → 边界填零（默认）
            'edge'  → 使用邻近差分值填充边界（复制）

    返回：
        diff_signal: np.ndarray，shape 与 signal 相同
    """
    signal = np.asarray(signal)
    time_dim = time_dim if time_dim >= 0 else signal.ndim + time_dim
    S = signal.shape

    # 将时间维移到 axis=0，统一处理
    sig_t = np.moveaxis(signal, time_dim, 0)  # shape: [T, ...]

    T = sig_t.shape[0]
    diff = np.zeros_like(sig_t)


    # 仅对中间部分应用卷积式二阶差分
    diff[3:-3] = (
        sig_t[0:-6] 
        + sig_t[6:]
        + 2 * (sig_t[1:-5] + sig_t[5:-1])
        - (sig_t[2:-4] + sig_t[4:-2])
        - 4 * sig_t[3:-3]
    ) / 16.0


    # 边界处理（可选）
    if pad_mode == 'edge':
        diff[:3] = diff[3]
        diff[-3:] = diff[-4]

    # 还原维度顺序
    return np.moveaxis(diff, 0, time_dim)
