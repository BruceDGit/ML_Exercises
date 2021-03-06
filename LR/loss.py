import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 500
w0, w1 = np.meshgrid(np.linspace(0, 9, n), np.linspace(0, 3.5, n))

xs = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
ys = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

loss = np.zeros_like(w0)
for x, y in zip(xs, ys):
    loss += (w0 + w1*x -y) ** 2 / 2

mp.figure('loss function', facecolor='lightgray')
mp.title('loss function', fontsize=18)

ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zlabel('loss')
ax3d.plot_surface(w0, w1, loss, rstride=30, cstride=30, cmap='jet')

mp.show()