import numpy as np
import sklearn.linear_model as lm
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.metrics as sm
import matplotlib.pyplot as mp

# train data
train_x, train_y = np.loadtxt('../data/single.txt', delimiter=',', usecols=(0, 1), unpack=True)
train_x = train_x.reshape(-1, 1)
print(train_x.shape, train_y.shape)

# polyfit model
model = pl.make_pipeline(sp.PolynomialFeatures(8), lm.LinearRegression())
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)

# estimate
print(sm.mean_absolute_error(train_y, pred_train_y))
print(sm.mean_squared_error(train_y, pred_train_y))
print(sm.median_absolute_error(train_y, pred_train_y))
print('r2_score:', sm.r2_score(train_y, pred_train_y))

# real data plotting
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, color='dodgerblue', s=80, label='Samples')

# test data
px = np.linspace(train_x.min(), train_x.max(), 1000)
py = model.predict(px.reshape(-1, 1))

# regression line
mp.plot(px, py, color='orangered', label='Regression Line')

mp.legend()
mp.show()













