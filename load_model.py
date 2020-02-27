import numpy as np
import pickle

# test data
test_x = np.linspace(1, 8, 20).reshape(-1, 1)
print(test_x.shape)

# load model
with open('models/linear.model', 'rb') as f:
    model = pickle.load(f)

# predict and print
pred_test_y = model.predict(test_x)
print(np.column_stack((test_x, pred_test_y)))