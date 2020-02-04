"""
    独热编码
"""
import numpy as np
import sklearn.preprocessing as sp


# raw_samples = np.array([
#     [17., 100., 4000],
#     [20., 80., 5000],
#     [23., 75., 5500]])
#
# ohe = sp.OneHotEncoder(sparse=False, dtype=int)
#
# ohe_dict = ohe.fit(raw_samples)
# ohe_samples = ohe_dict.transform(raw_samples)
# print(ohe_samples)
#
# ohe_samples = ohe.fit_transform(raw_samples)
# print(ohe_samples)

# ---------------

data = np.mat('1 3 2; 7 5 4; 1 8 6; 7 3 9')
print(type(data))

ohe = sp.OneHotEncoder()
r = ohe.fit_transform(data)
print(r)  # 稀疏矩阵, 只显示矩阵中那个位置不为0
print(r.toarray())