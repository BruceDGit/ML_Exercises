import numpy as np
import sklearn.preprocessing as sp
import matplotlib.pyplot as mp

xs = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
ys = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

times = 1000
w0, w1 = 1, 1
lrate = 0.01

for i in range(1, times+1):
    # todo?
    d0 = (w0 + w1 * xs - ys).sum()
    d1 = (xs * (w0 + w1 * xs - ys)).sum()
    w0 = w0 - lrate * d0
    w1 = w1 - lrate * d1

mp.figure("Linear Regression", facecolor='lightgray')
mp.title('Linear Rgression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(xs, ys, s=80, color='dodgerblue', label='Samples')

pred_y = w0 + w1 * xs
mp.plot(xs, pred_y, color='orangered', label="Regression Line")
mp.legend()
mp.show()