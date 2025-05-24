import numpy as np

def beamforming_2d(
    ## TI 6843AOP
    radar_data_seg,          
    Tx_pos=np.array([[2, 0], [0, 2], [2, 2]]),          
    Rx_pos=np.array([[0, 1], [0, 0], [1, 1], [1, 0]]), 
    win_2d=None,             
    el_range=np.arange(-45, 46, 3),
    az_range=np.arange(-45, 46, 3)
):
    """
    2D 波束成形（俯仰 x 方位）

    参数：
        radar_data_seg: ndarray, shape [time_slice, range bin, virtual_channels]
        Tx_pos: 发射阵列位置 [num_tx, 2] ! The position is set to relative.
        Rx_pos: 接收阵列位置 [num_rx, 2] ! The position is set to relative.
        win_2d: 可选的窗口函数（如 Hamming), shape [num_tx, num_rx]
        el_range: 俯仰角度数组（单位：度）
        az_range: 方位角度数组（单位：度）

    返回：
        DBF_data: ndarray, shape [time, len(el_range), len(az_range)]
    """
    num_tx = Tx_pos.shape[0]
    num_rx = Rx_pos.shape[0]
    num_virtual = num_tx * num_rx

    # ========== 构造虚拟阵列 ========== #
    virtual_pos = np.zeros((num_virtual, 2))
    window_pos = np.ones((num_virtual,), dtype=np.float32)

    idx = 0
    for t in range(num_tx):
        for r in range(num_rx):
            virtual_pos[idx, :] = Tx_pos[t, :] + Rx_pos[r, :]
            if win_2d is not None:
                window_pos[idx] = win_2d[t, r]
            idx += 1

    # 应用窗函数到数据：乘以虚拟通道方向上的权重
    if win_2d is not None:
        radar_data_seg = radar_data_seg * window_pos[None, :]

    # ========== 波束成形（2D角度） ========== #

    DBF_data = np.zeros((radar_data_seg.shape[0], len(el_range), len(az_range)), dtype=np.complex64)
    
    el = np.radians(el_range)  # shape: [E]
    az = np.radians(az_range)  # shape: [A]

    # ===== 计算导向矢量矩阵 [C, E, A] =====
    # broadcast: [C,1,1], [1,E,1], [1,1,A]
    x = virtual_pos[:, 0][:, None, None]  # [C,1,1]
    y = virtual_pos[:, 1][:, None, None]  # [C,1,1]
    el = el[None, :, None]                # [1,E,1]
    az = az[None, None, :]                # [1,1,A]
    
    phase = np.pi  * (x * np.cos(el) * np.sin(az) + y * np.sin(el))  # [C,E,A]
    steer_matrix = np.exp(1j * phase).astype(np.complex64)  # [C,E,A]
    DBF_data = np.einsum('trc,cea->trea', radar_data_seg, steer_matrix)

    return DBF_data


