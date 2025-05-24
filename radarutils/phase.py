import numpy as np

def circle_fit(IQ_data):
    """
    对复数 IQ 数据进行圆拟合，相位中心校正（去除 DC 偏移）

    参数：
        IQ_data: ndarray, shape [N,] 或 [T,]，复数数组

    返回：
        corrected_data: 同 shape，校正后的复数数据
        center: 拟合出的复数圆心（可选返回）
    """
    # IQ_data = np.asarray(IQ_data).flatten()
    
    # 构造拟合矩阵
    H = np.column_stack([
        2 * np.real(IQ_data),
        2 * np.imag(IQ_data),
        np.ones_like(IQ_data)
    ])
    Y = np.abs(IQ_data) ** 2

    # 最小二乘解圆心
    # Center_circ = (H.T @ H)^-1 @ H.T @ Y
    # pseudo_inv = np.linalg.pinv(H.T @ H) @ H.T
    center_params = np.linalg.pinv(H.T @ H) @ H.T @ Y

    # 拟合圆心
    center_complex = center_params[0] + 1j * center_params[1]

    # 去除圆心：校正信号
    corrected_data = IQ_data - center_complex

    return corrected_data, center_complex
