import numpy as np
from radarutils.filters import differentiator_2order


inputs = np.array([[1, 2, 4, 1, 5, 6., 7, 8, 9, 11 , 12, 13, 12, 11, 12 , 11]]).T
# inputs = np.random.rand(100,2)
print(inputs)
print(inputs.shape)
results =  differentiator_2order(inputs)
print(results)