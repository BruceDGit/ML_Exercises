"""
    均值移除（标准化）
"""
import numpy as np
import sklearn.preprocessing as sp


raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])
print(raw_samples)

std_samples = sp.scale(raw_samples)
print(std_samples)
print(std_samples.mean(axis=0))  # 均为0
print(std_samples.std(axis=0))   # 均为1
