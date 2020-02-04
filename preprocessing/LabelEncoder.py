"""
    标签编码
"""
import numpy as np
import sklearn.preprocessing as sp


raw_samples = np.array([
    'audi', 'ford', 'audi', 'toyota',
    'ford', 'bmw', 'toyota', 'ford',
    'audi'])
print(raw_samples)

lbe = sp.LabelEncoder()

lbe_samples = lbe.fit_transform(raw_samples)
print(lbe_samples)

# 假设预测结果为pred
pred = [0, 1, 2, 3, 3, 2, 3, 2, 2, 1]
pred_text = lbe.inverse_transform(pred)
print(pred_text)
