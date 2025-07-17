import numpy as np







def read_ecg_from_txt(file_path, start_time, end_time):
    """
    从 ECG txt 文件中读取数据，并根据时间范围提取信号段。

    参数：
        file_path: str, ECG 数据文件路径
        start_time: float, 开始时间（秒，POSIX）
        end_time: float, 结束时间（秒，POSIX）

    返回：
        ecg_values: np.ndarray, ECG 电压值
        ecg_timestamps: np.ndarray, 对应的 POSIX 时间戳（秒）
    """
    # 读取整张表：两列 [timestamp_or_value, ecg_value_or_marker]
    data = np.loadtxt(file_path)
    # 时间列
    rough_time = data[:, 0]

    # 找到起止索引
    start_idx = np.searchsorted(rough_time, start_time, side='left')
    end_idx = np.searchsorted(rough_time, end_time, side='left')
    mixed_values = data[start_idx:end_idx, 1]


    ecg_values = []
    ecg_timestamps = []

    current_time = None

    for val in mixed_values:
        if val > 15000:  # 类似 MATLAB 的 16 位时间戳判断
            current_time = val
        elif current_time is not None:
            ecg_timestamps.append(current_time)
            ecg_values.append(val)

    ecg_values = np.array(ecg_values, dtype=np.float32)
    ecg_timestamps = np.array(ecg_timestamps, dtype=np.float64)
    ecg_timestamps -= ecg_timestamps[0]  # 从 0 开始

    return ecg_values, ecg_timestamps


def replace_outliers_by_row_mean(arr, threshold=1.2):
    arr = arr.copy()  # 避免修改原数组
    for i in range(arr.shape[0]):
        row = arr[i]
        row_mean = np.mean(row)
        # 标记异常值（大于该行均值的 1.8 倍）
        mask = row > (row_mean * threshold)
        # 取非异常值的平均作为替换值
        if np.any(~mask):
            replacement = np.mean(row[~mask])
            row[mask] = replacement
            arr[i] = row
    return arr

def replace_outliers_by_row_mean_low(arr, threshold=0.8):
    arr = arr.copy()  # 避免修改原数组
    for i in range(arr.shape[0]):
        row = arr[i]
        row_mean = np.mean(row)
        # 标记异常值（大于该行均值的 1.8 倍）
        mask = row < (row_mean * threshold)
        # 取非异常值的平均作为替换值
        if np.any(~mask):
            replacement = np.mean(row[~mask])
            row[mask] = replacement
            arr[i] = row
    return arr


def RTIBIgen(peaks,sig_length,peaks_score=None):

    RTIBI = np.zeros((1,sig_length))

    if peaks_score is not None:
        confidence = np.ones((1,sig_length))

    if len(peaks) > 2:
        # 第一个区间赋值
        RTIBI[0,:peaks[0]] = peaks[1] - peaks[0]
        if peaks_score is not None:
            confidence[0,:peaks[0]] = peaks_score[0]
        # 遍历峰值数组的每对相邻值，计算区间内的值
        for IBI_index in range(len(peaks) - 1):
            start_idx = peaks[IBI_index]
            end_idx = peaks[IBI_index + 1]
            RTIBI[0,start_idx:end_idx] = peaks[IBI_index + 1] - peaks[IBI_index]
            if peaks_score is not None:
                confidence[0,start_idx:end_idx] = peaks_score[IBI_index]
        # 最后一个区间赋值
        RTIBI[0,peaks[-1]:] = peaks[-1] - peaks[-2]
        if peaks_score is not None:
            confidence[0,peaks[-1]:] = peaks_score[-1]
        # 将结果乘以 10
        RTIBI *= 10
    RTIBI[RTIBI>1500] = 1500
    RTIBI[RTIBI<400] = 400
    RTIBI = replace_outliers_by_row_mean(RTIBI,1.2)
    RTIBI = replace_outliers_by_row_mean_low(RTIBI,0.8)

    if peaks_score is not None:
        return RTIBI,confidence
    else:
        return RTIBI
