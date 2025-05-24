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
