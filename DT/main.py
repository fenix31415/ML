import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


FOLDER_NAME = "DT_csv/"
MAX_DEPTH = 50


def draw_fun_abstract(bounds, f, ax_xlabel="", ax_ylabel="", title=""):
    left, right, step_size = bounds
    step_count = int(np.floor((right - left) / step_size))
    # right = left + step_count * step_size

    plot_x = [left + i * step_size for i in range(step_count)]
    plot_y = [f(x) for x in plot_x]

    # fig, ax = plt.subplots()
    # ax.set_xlabel(ax_xlabel)
    # ax.set_ylabel(ax_ylabel)
    # plt.title(title)
    # plt.plot(plot_x, plot_y)

    return plot_x, plot_y


class RandomForest:
    def __init__(self, train_x, train_y, trees_number):
        self.X = train_x
        self.Y = train_y
        self.classes = max(train_y)
        self.forest = [self._build_tree() for _ in range(trees_number)]

    def _build_tree(self):
        # inds = np.random.randint(len(self.Y) - 1, size=int(np.math.sqrt(len(self.Y))))
        inds = np.random.randint(len(self.Y) - 1, size=int(0.8*len(self.Y)))
        _X = [self.X[i] for i in inds]
        _Y = [self.Y[i] for i in inds]
        return train_tree([_X, _Y], "gini", "best", 100000000)

    def predict(self, xs):
        classes = np.zeros((len(xs), self.classes + 1))
        for tree in self.forest:
            anses_tree_i = tree.predict(xs)
            for test_num in range(len(xs)):
                classes[test_num][anses_tree_i[test_num]] += 1
        return np.argmax(classes, axis=1)

    def predict_(self, x):
        classes = np.zeros(self.classes + 1)
        for tree in self.forest:
            classes[tree.predict([x])[0]] += 1
        return np.argmax(classes)


def read_data(filename):
    dataframe = pd.read_csv(filename)
    x = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    return x.to_numpy(), y.to_numpy()


def read_dataset(dataset_number, dataset_type):
    return read_data(FOLDER_NAME + dataset_number + dataset_type + ".csv")


def train_tree(train, criterion, splitter, max_depth):
    t = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
    train_x = train[0]
    train_y = train[1]
    t.fit(train_x, train_y)
    return t


def do_test(test, t):
    test_x = test[0]
    test_y = test[1]
    anses = t.predict(test_x)
    return accuracy_score(test_y, anses)


def find_hyperparams(train, test, out=True):
    criterions = ["gini", "entropy"]
    splitters = ["best", "random"]
    max_depths = [i for i in range(1, MAX_DEPTH)]

    max_ans = 0
    max_params = []
    for criterion in criterions:
        for splitter in splitters:
            for max_depth in max_depths:
                ans = do_test(test, train_tree(train, criterion, splitter, max_depth))
                if ans > max_ans:
                    max_ans = ans
                    max_params = [criterion, splitter, max_depth]
                    if out:
                        print("max has updated", max_ans, max_params)

    if out:
        print(max_params, "max_ans: ", max_ans)

    return max_params, max_ans


def draw_(bounds, test, train):
    def f(t):
        return do_test(test, train_tree(train, "gini", "best", t))

    return draw_fun_abstract(bounds, f)


def draw(bounds, n):
    train = read_dataset(n, "_train")
    test = read_dataset(n, "_test")

    x_test, y_test = draw_(bounds, test, train)
    x_train, y_train = draw_(bounds, train, train)
    plt.plot(x_train, y_train, label="train")
    plt.plot(x_test, y_test, label="test")
    plt.legend()
    plt.xlabel("depth")
    plt.ylabel("accuracy")
    plt.title("dataset {}".format(n))
    plt.show()


def do_tree_all():
    tests = 21
    min_depth = 100000
    max_depth = 0
    min_ind = 0
    max_ind = 0
    for test in range(1, tests + 1):
        n = str(test).zfill(2)
        train = read_dataset(n, "_train")
        test = read_dataset(n, "_test")
        params, acc = find_hyperparams(train, test, False)
        print(n, params, acc)
        cur_depth = params[2]
        if cur_depth > max_depth:
            max_depth = cur_depth
            max_ind = n
        if cur_depth < min_depth:
            min_depth = cur_depth
            min_ind = n

    draw([1, MAX_DEPTH, 1], max_ind)
    draw([1, MAX_DEPTH, 1], min_ind)


def do_forest(n):
    train_x, train_y = read_dataset(n, "_train")
    test_x, test_y = read_dataset(n, "_test")
    f = RandomForest(train_x, train_y, 100)
    acc_test = do_test([test_x, test_y], f)
    acc_train = do_test([train_x, train_y], f)
    print("dataset: {}, accuracy on train = {}, on test = {}".format(n, acc_train, acc_test))


def do_forest_all():
    tests = 21
    for test in range(1, tests + 1):
        n = str(test).zfill(2)
        do_forest(n)


do_tree_all()
# draw([1, MAX_DEPTH, 1], "03")
# draw([1, MAX_DEPTH, 1], "21")
# do_forest_all()
