import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def read_data(name):
    dataframe = pd.read_csv(name)
    dataset_y = np.array([1 if dataframe.to_numpy()[:, -1][i] == "P" else -1 for i in range(len(dataframe.to_numpy()))])
    dataframe.drop(columns=['class'], inplace=True)
    min_max = get_minmax(dataframe.to_numpy())
    dataset_x = normalize(dataframe.to_numpy(), min_max)
    return dataset_x, dataset_y


def get_minmax(data):
    return [[data[:, i].min(), data[:, i].max()] for i in range(len(data[0]))]


def normalize(data, minmax):
    for row in data:
        for i in range(len(row)):
            num = row[i] - minmax[i][0]
            den = minmax[i][1] - minmax[i][0]
            row[i] = num / den if den != 0 else 0
    return data


def train_tree(train, max_depth, w, criterion="gini", splitter="best"):
    t = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, splitter=splitter)
    train_x = train[0]
    train_y = train[1]
    t.fit(train_x, train_y, sample_weight=w)
    return t


class AdaBoost:
    MAX_DEPTH = 3
    CRITERION = "gini"
    SPLITTER = "best"

    def __init__(self, train):
        self.alphas = []
        self.trees = []
        self.X = train[0]
        self.Y = train[1]
        self.w = [1.0 / len(self.X)] * len(self.X)
        self.step_number = 0

    def step(self):
        a_t = train_tree([self.X, self.Y], AdaBoost.MAX_DEPTH, self.w, AdaBoost.CRITERION, AdaBoost.SPLITTER)
        self.trees.append(a_t)
        anses = a_t.predict(self.X)
        n_t = sum([(self.w[i] if anses[i] != self.Y[i] else 0) for i in range(len(self.X))])
        b_t = 0.5 * np.log((1 - n_t) / n_t)
        self.alphas.append(b_t)
        for i in range(len(self.X)):
            self.w[i] *= np.exp(-b_t * self.Y[i] * anses[i])
        w_0 = sum(self.w)
        for i in range(len(self.X)):
            self.w[i] /= w_0
        self.step_number += 1

    def classify(self, x):
        return np.sign(sum([self.alphas[i] * self.trees[i].predict([x])[0] for i in range(len(self.alphas))]))


def cv_get_folds(dataset_x, dataset_y, pack_size):
    ind = 0
    folds = []
    while ind < len(dataset_y):
        cur_pack_size = min(pack_size, len(dataset_y) - ind)
        to_delete = range(ind, ind + cur_pack_size)
        newdataset_x_train = np.delete(dataset_x, to_delete, 0)
        newdataset_y_train = np.delete(dataset_y, to_delete, 0)
        newdataset_x_test, newdataset_y_test = [], []
        for i in range(ind, ind + cur_pack_size):
            newdataset_x_test.append(dataset_x[i])
            newdataset_y_test.append(dataset_y[i])
        ind += cur_pack_size
        folds.append([[newdataset_x_train, newdataset_y_train], [newdataset_x_test, newdataset_y_test]])
    return folds


def cv(folds, ts):
    my_anses = []
    riganses = []
    for fold_num in range(len(folds)):
        _, test = folds[fold_num]
        for i in range(len(test[0])):
            riganses.append(test[1][i])
            my_anses.append(ts[fold_num].classify(test[0][i]))
    return accuracy_score(riganses, my_anses)


def draw_cv_(steps, dataset_x, dataset_y, k=2):
    xs = [i for i in range(1, steps + 1)]
    ys = []

    folds = cv_get_folds(dataset_x, dataset_y, k)
    ts = [AdaBoost(folds[i][0]) for i in range(len(folds))]
    for step in range(1, steps + 1):
        for t in ts:
            t.step()

        ys.append(cv(folds, ts))

    plt.plot(xs, ys)


def draw_cv(filename):
    steps = 100
    dataset_x, dataset_y = read_data(filename)
    draw_cv_(steps, dataset_x, dataset_y)
    plt.title("Dataset: {}".format(filename.rstrip(".csv")))
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.show()


def draw_cvs():
    for filename in ["chips.csv", "geyser.csv"]:
        draw_cv(filename)


def draw_dataset(dataset_x, dataset_y):
    xs_all = dataset_x[:, 0]
    ys_all = dataset_x[:, 1]
    xs = [[], []]
    ys = [[], []]
    for i in range(len(xs_all)):
        ind = 0 if dataset_y[i] == 1 else 1
        xs[ind].append(xs_all[i])
        ys[ind].append(ys_all[i])

    plt.scatter(xs[0], ys[0], marker='+')
    plt.scatter(xs[1], ys[1], marker='_')


def _draw(dataset_x, dataset_y, t):
    squares_per_line = 200
    sqrrng = range(squares_per_line)
    sqr_len = 1.0 / squares_per_line
    rect_coord = [x * sqr_len + sqr_len / 2.0 for x in sqrrng]

    data = [[t.classify([rect_coord[x], rect_coord[y]]) for y in sqrrng] for x in sqrrng]

    def get_rgb(x):
        return [0, 0, 0] if x == 1 else [255, 255, 255]

    rgb_data = [[get_rgb(data[x][y]) for x in sqrrng] for y in sqrrng]
    fig, ax = plt.subplots()
    # ax.imshow(data, interpolation='nearest', origin='lower', cmap=cmap, norm=norm,  extent=[0, 1, 0, 1])
    ax.imshow(rgb_data, interpolation='quadric', extent=[0, 1, 0, 1], origin='lower')

    draw_dataset(dataset_x, dataset_y)


def draw(filename):
    dataset_x, dataset_y = read_data(filename)
    t = AdaBoost([dataset_x, dataset_y])
    for i in range(100):
        t.step()
        if t.step_number in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            _draw(dataset_x, dataset_y, t)
            plt.title("Dataset: {}, steps: {}".format(filename.rstrip(".csv"), t.step_number))
            plt.show()


def draw_all():
    for filename in ["chips.csv", "geyser.csv"]:
        draw(filename)


draw_cvs()
# draw_all()
