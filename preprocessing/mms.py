"""
    范围缩放
"""
import numpy as np
import sklearn.preprocessing as sp


raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])
print(raw_samples)

mms_samples = raw_samples.copy()
for col in mms_samples.T:
    col_min = col.min()
    col_max = col.max()
    a = np.array([
        [col_min, 1],
        [col_max, 1]
    ])
    b = np.array([0, 1])
    x = np.linalg.solve(a, b)  # 返回方程组ax=b的解，shape与b相同
    col *= x[0]
    col += x[1]
print(mms_samples)

mms = sp.MinMaxScaler(feature_range=(0, 1))
mms_samples = mms.fit_transform(raw_samples)
print(mms_samples)