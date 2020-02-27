import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import pickle

# data
train_x, train_y = np.loadtxt('data/single.txt', delimiter=',', usecols=(0, 1), unpack=True)
train_x = train_x.reshape(-1, 1)

# train
model = lm.LinearRegression()
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)

# estimate
print(sm.mean_absolute_error(train_y, pred_train_y))
print(sm.mean_squared_error(train_y, pred_train_y))
print(sm.median_absolute_error(train_y, pred_train_y))

print(sm.r2_score(train_y, pred_train_y))

# save model
with open('models/linear.model', 'wb') as f:
    pickle.dump(model, f)

print('save "linear.model" success')
