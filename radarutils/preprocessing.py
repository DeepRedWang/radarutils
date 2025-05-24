import numpy as np

def normalize(x, axis=None, method='zscore'):
    if method == 'zscore':
        return (x - np.mean(x, axis=axis, keepdims=True)) / np.std(x, axis=axis, keepdims=True)
    elif method == 'minmax':
        return (x - np.min(x, axis=axis, keepdims=True)) / (np.max(x, axis=axis, keepdims=True) - np.min(x, axis=axis, keepdims=True))
    elif method == 'l2':
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / norm
    else:
        raise ValueError("Unknown normalization method")