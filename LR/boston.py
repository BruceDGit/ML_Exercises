import numpy as np
import sklearn.datasets as sd
import sklearn.tree as st
import sklearn.utils as su
import sklearn.metrics as sm
import sklearn.linear_model as lm


class RegressionModel:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.train_x = self.test_x = None
        self.train_y = self.test_y = None
        self.shuffle_data()

    def shuffle_data(self):
        x, y = su.shuffle(self.xs, self.ys, random_state=7)
        train_size = int(len(x) * 0.8)
        self.train_x, self.test_x = x[:train_size], x[train_size:]
        self.train_y, self.test_y = y[:train_size], y[train_size:]

    def decision_tree_regress(self):
        # self.shuffle_data()
        model = st.DecisionTreeRegressor(max_depth=4)
        model.fit(self.train_x, self.train_y)
        # predict
        pred_test_y = model.predict(self.test_x)
        # estimate
        print(sm.r2_score(self.test_y, pred_test_y))
        print(sm.mean_absolute_error(self.test_y, pred_test_y))

    def linear_regress(self):
        # self.shuffle_data()
        model = lm.LinearRegression()
        model.fit(self.train_x, self.train_y)
        # predict
        pred_test_y = model.predict(self.test_x)
        # estimate
        print(sm.r2_score(self.test_y, pred_test_y))
        print(sm.mean_absolute_error(self.test_y, pred_test_y))


if __name__ == '__main__':
    boston = sd.load_boston()
    # print(boston.data.shape, boston.data[0])
    # print(boston.target.shape, boston.target[0])
    # print(boston.feature_names)
    rm = RegressionModel(boston.data, boston.target)
    rm.decision_tree_regress()
    print('-' * 45)
    rm.linear_regress()