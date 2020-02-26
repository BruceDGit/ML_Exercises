import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

train_x, train_y = np.loadtxt(
    '../data/single.txt',
    delimiter=',',
    usecols=(0, 1),
    unpack=True)
train_x = train_x.reshape(-1, 1)
print(train_x.shape, train_y.shape)

model = lm.LinearRegression()
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)

print(sm.mean_absolute_error(train_y, pred_train_y))
print(sm.mean_squared_error(train_y, pred_train_y))
print(sm.median_absolute_error(train_y, pred_train_y))

print('r2_score:', sm.r2_score(train_y, pred_train_y))

mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, color='dodgerblue', s=80, label='Samples')

mp.plot(train_x, pred_train_y, color='orangered', label='Regression Line')

mp.legend()
mp.show()