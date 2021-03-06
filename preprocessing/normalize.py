"""
    归一化
"""
import numpy as np
import sklearn.preprocessing as sp


raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])
print(raw_samples)

nor_samples = raw_samples.copy()
for row in nor_samples:
    row /= abs(row).sum()
print(nor_samples)
print(abs(nor_samples).sum(axis=1))

# API
nor_samples = sp.normalize(raw_samples, norm='l1')
print(nor_samples)
print(abs(nor_samples).sum(axis=1))