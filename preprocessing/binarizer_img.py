"""
    二值化
"""
import numpy as np
import sklearn.preprocessing as sp
import scipy.misc as sm
import matplotlib.pyplot as mp


img = sm.imread("../data/lily.jpg", True)
print(img)

# normal
mp.subplot(121)
mp.imshow(img, cmap='gray')
mp.tight_layout()

# binarized
bin = sp.Binarizer(threshold=127)
img2 = bin.transform(img)
print(img2)

mp.subplot(122)
mp.imshow(img2, cmap='gray')
mp.tight_layout()

mp.show()
