import numpy as np

# class Cal_HRV:

#     def __init__(self,):
#         pass


def cal_HRV(est_IBI_seq, GT_IBI_seq):  

    # 降采样之后再计算
    est_RMSSD = np.sqrt(np.mean(np.diff(est_IBI_seq,axis=-1)**2,axis=-1))
    GT_RMSSD = np.sqrt(np.mean(np.diff(GT_IBI_seq,axis=-1)**2,axis=-1))

    est_SDRR = np.sqrt(np.mean((est_IBI_seq-np.mean(est_IBI_seq,axis=-1,keepdims=True))**2,axis=-1))
    GT_SDRR = np.sqrt(np.mean((GT_IBI_seq-np.mean(GT_IBI_seq,axis=-1,keepdims=True))**2,axis=-1))

    est_pNN50 = np.sum(np.diff(est_IBI_seq,axis=-1)>50,axis=-1)/est_IBI_seq.shape[-1]
    GT_pNN50 = np.sum(np.diff(GT_IBI_seq,axis=-1)>50,axis=-1)/GT_IBI_seq.shape[-1]

    error_RMSSD = np.abs(est_RMSSD-GT_RMSSD)
    error_SDRR = np.abs(est_SDRR-GT_SDRR)
    error_pNN50 = np.abs(est_pNN50-GT_pNN50)

    return  error_RMSSD, error_SDRR, error_pNN50

