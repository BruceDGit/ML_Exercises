import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# data
train_x, train_y = np.loadtxt('../data/abnormal.txt', delimiter=',', usecols=(0, 1), unpack=True)
train_x = train_x.reshape(-1, 1)
print(train_x.shape, train_y.shape)

# lr model
model = lm.LinearRegression()
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)

mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, color='dodgerblue', s=80, label='Samples')
mp.plot(train_x, pred_train_y, color='orangered', label='Regression Line')

# ridge model
model = lm.Ridge(150, fit_intercept=True, max_iter=1000)
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)
mp.plot(train_x, pred_train_y, color='limegreen', label='Ridge Regression Line')

mp.legend()
mp.show()
