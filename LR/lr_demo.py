import numpy as np
import sklearn.preprocessing as sp
import matplotlib.pyplot as mp

xs = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
ys = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

times = 1000
w0, w1, losses, epoches = [1], [1], [], []
lrate = 0.2

for i in range(1, times+1):
    loss = ((w0[-1] + w1[-1]*xs - ys)**2).sum() / 2
    losses.append(loss)
    epoches.append(i)
    print('{:4}> w0={:.6f}, w1={:.6f}, loss={:.6f}'.format(i, w0[-1], w1[-1], loss))
    d0 = (w0[-1] + w1[-1]*xs - ys).sum()
    d1 = (xs * (w0[-1] + w1[-1]*xs - ys)).sum()
    w0.append(w0[-1] - lrate * d0)
    w1.append(w1[-1] - lrate * d1)

mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(xs, ys, s=80, color='dodgerblue', label='Samples')
pred_y = w0[-1] + w1[-1] * xs
mp.plot(xs, pred_y, color='orangered', label='Regression Line')
mp.legend()

# subplot
w0 = w0[:-1]
w1 = w1[:-1]

mp.figure('Training Progress', facecolor='lightgray')
mp.subplot(311)
mp.title('Training Progress', fontsize=20)
mp.ylabel('w0', fontsize=14)
mp.gca().xaxis.set_major_locator(mp.MultipleLocator(100))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(epoches, w0, c='dodgerblue', label='w0')
mp.legend()

mp.subplot(312)
mp.ylabel('w1', fontsize=14)
mp.gca().xaxis.set_major_locator(mp.MultipleLocator(100))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(epoches, w1, c='dodgerblue', label='w1')
mp.legend()

mp.subplot(313)
mp.ylabel('loss', fontsize=14)
mp.gca().xaxis.set_major_locator(mp.MultipleLocator(100))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(epoches, losses, c='dodgerblue', label='loss')
mp.legend()

# todo Gradient Descent Line

mp.show()













