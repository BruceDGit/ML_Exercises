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

# Linear Regression
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(xs, ys, s=80, color='dodgerblue', label='Samples')
pred_y = w0[-1] + w1[-1] * xs
mp.plot(xs, pred_y, color='orangered', label='Regression Line')
mp.legend()

# subplot Training Progress
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

# Gradient Descent Line
import mpl_toolkits.mplot3d as axes3d

grid_w0, grid_w1 = np.meshgrid(
    np.linspace(0, 9, 500),
    np.linspace(0, 5, 500))

grid_loss = np.zeros_like(grid_w0)
for x, y in zip(xs, ys):
    grid_loss += ((grid_w0 + x*grid_w1 - y) ** 2) / 2

mp.figure('Loss Function')
ax = mp.gca(projection='3d')
mp.title('Loss Function', fontsize=20)
ax.set_xlabel('w0', fontsize=14)
ax.set_ylabel('w1', fontsize=14)
ax.set_zlabel('loss', fontsize=14)
ax.plot_surface(grid_w0, grid_w1, grid_loss, rstride=20, cstride=20, cmap='jet')
ax.plot(w0, w1, losses, 'o-', c='orangered', label='BGD')
mp.legend()

# Gradient Descent Line2
mp.figure('Batch Gradient Descent', facecolor='lightgray')
mp.title('Batch Gradient Descent', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.contourf(grid_w0, grid_w1, grid_loss, 10, cmap='jet')
cntr = mp.contour(grid_w0, grid_w1, grid_loss, 10, colors='black', linewidths=0.5)
mp.clabel(cntr, inline_spacing=0.1, fmt='%.2f', fontsize=8)
mp.plot(w0, w1, 'o-', c='orangered', label='BGD')
mp.legend()


mp.show()













