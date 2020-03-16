import sklearn.datasets as sd
import sklearn.tree as st
import sklearn.utils as su
import sklearn.metrics as sm
import sklearn.ensemble as se


class Adaboost:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.train_x = self.test_x = None
        self.train_y = self.test_y = None
        self.shuffle_data()
        self.model = None

    def shuffle_data(self):
        x, y = su.shuffle(self.xs, self.ys, random_state=7)
        train_size = int(len(x) * 0.8)
        self.train_x, self.test_x = x[:train_size], x[train_size:]
        self.train_y, self.test_y = y[:train_size], y[train_size:]

    def train(self):
        self.model = st.DecisionTreeRegressor(max_depth=4)
        self.model = se.AdaBoostRegressor(self.model, n_estimators=400, random_state=7)
        self.model.fit(self.train_x, self.train_y)

    def test(self):
        pred_test_y = self.model.predict(self.test_x)
        print(sm.r2_score(self.test_y, pred_test_y))
        print(sm.mean_absolute_error(self.test_y, pred_test_y))

    def decision_tree_regress(self):
        model = st.DecisionTreeRegressor(max_depth=4)
        model.fit(self.train_x, self.train_y)
        # predict
        pred_test_y = model.predict(self.test_x)
        # estimate
        print(sm.r2_score(self.test_y, pred_test_y))
        print(sm.mean_absolute_error(self.test_y, pred_test_y))


if __name__ == '__main__':
    boston = sd.load_boston()
    print(boston.data.shape, boston.data[0])
    print(boston.target.shape, boston.target[0])
    print(boston.feature_names)

    adb = Adaboost(boston.data, boston.target)
    adb.train()
    adb.test()

    print('-'*45)
    adb.decision_tree_regress()