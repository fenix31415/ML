from functools import partial

import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt


class Kernels:
    @staticmethod
    def linear_fabric(shift=0):
        def prod(a, b):
            return np.asarray(a).dot(np.asarray(b)) + shift

        return Kernels(prod, "linear kernel" + ("" if shift == 0 else ", shift = {}".format(shift)))

    @staticmethod
    def poly_fabric(d, r=0):
        def prod(a, b):
            return (np.asarray(a).dot(np.asarray(b).T) + r) ** d

        return Kernels(prod, "poly kernel, deg = {}".format(d) + ("" if r == 0 else ", r = {}".format(r)))

    @staticmethod
    def gauss_fabric(beta):
        def prod(a, b):
            return math.exp(-beta * np.linalg.norm(a - b) ** 2)

        return Kernels(prod, "gauss kernel, beta = {}".format(beta))

    def __init__(self, k, name):
        self.description = name
        self.kern = k

    def __call__(self, x_i, x_j):
        return self.kern(x_i, x_j)

    def __str__(self):
        return self.description


def cheety_kernel2(a, b):
    x1 = a[0]
    x2 = b[0]
    y1 = a[1]
    y2 = b[1]
    return x1 * x2 + y1 * y2 + ((x1 - 0.5) ** 2 + (y1 - 0.5) ** 2) * ((x2 - 0.5) ** 2 + (y2 - 0.5) ** 2)


def cheety_kernel1(a, b):
    x1 = a[0]
    x2 = b[0]
    y1 = a[1]
    y2 = b[1]
    return x1 * x2 + y1 * y2 + (x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2)


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


def get_accuracy(my_anses, anses):
    return (len(my_anses) - np.count_nonzero(np.asarray(my_anses) - np.asarray(anses))) / len(my_anses)


def pr(v, a, c):
    k = v.dot(a) / np.linalg.norm(a) ** 2
    for i in range(len(v)):
        if abs(v[i]) > EPS and abs(v[i] - c) > EPS:
            v[i] -= k * a[i]
    return v


def is_in_cube(c, v):
    return min(v) > -EPS and max(v) < c + EPS


def fix_newlambda(v):
    eps = 1e-13
    for i in range(len(v)):
        if abs(v[i]) < eps:
            v[i] = 0
            print("fixed!", i)
        if abs(10 - v[i]) < eps:
            v[i] = 10
            print("fixed!", i)
    return v


def fix_pr(v, a, c):
    return pr(v, a, c)


def fun(lambdas, y):
    ans = 0
    for i in range(len(lambdas)):
        ans -= lambdas[i]
        for j in range(len(lambdas)):
            ans += lambdas[i] * lambdas[j] * y[i] * y[j] * mem[i][j]
    return ans


def fit(v, c):
    for i in range(len(v)):
        v[i] = min(max(v[i], 0), c)
    return v


def find_lambda_(train_x, train_y, kern, c):
    max_iters = 50

    size = len(train_y)
    size_ran = range(size)
    global mem
    mem = [[kern(train_x[i], train_x[j]) for j in size_ran] for i in size_ran]
    lambdas = np.zeros(size)

    step_number = 1
    while step_number < max_iters:
        mu = 0.001
        delta = np.zeros(size)
        for k in size_ran:
            summ = 0
            for i in size_ran:
                summ += lambdas[i] * train_y[i] * train_y[k] * mem[i][k]
            delta[k] = mu * (summ - 1)
        lambdas = pr(fit(lambdas - delta, c), train_y, c)
        step_number += 1

    # finding w_0:
    support_ind = 0
    while (support_ind < len(lambdas)) and (abs(lambdas[support_ind]) < EPS or abs(lambdas[support_ind] - c) < EPS):
        support_ind += 1
    if support_ind >= len(lambdas):
        print("Gg")
        support_ind = 0
    sum_for_support = get_support_sum(train_x, train_y, lambdas, kern, support_ind, True)
    b = train_y[support_ind] - sum_for_support
    return lambdas, b


def find_lambda(train_x, train_y, kern, c):  # fills matrix mem [size, size]
    max_iters = 50

    size = len(train_y)
    size_ran = range(size)
    global mem
    mem = [[kern(train_x[i], train_x[j]) for j in size_ran] for i in size_ran]
    lambdas = [c / 2.0] * size
    # lambdas = np.zeros(size)
    bounded = [False] * size

    mu = c
    koef = 0.9
    step_number = 0

    while (step_number < max_iters) and mu > 1e-15:
        step_number += 1
        delta = np.zeros(size)
        for k in size_ran:
            summ = 0
            for i in size_ran:
                summ += lambdas[i] * train_y[i] * train_y[k] * mem[i][k]
            if abs(lambdas[k] - c) > EPS:
                delta[k] = mu * (summ - 1)
            else:
                if not bounded[k]:
                    bounded[k] = True
                    # mu = c
                delta[k] = 0
                lambdas[k] = c
        newlambdas = fix_pr(lambdas - delta, train_y, c)
        if is_in_cube(c, newlambdas):
            lambdas = newlambdas
        else:
            mu *= koef

    # finding w_0:
    support_ind = 0
    while abs(lambdas[support_ind]) < EPS or abs(lambdas[support_ind] - c) < EPS:
        support_ind += 1
    sum_for_support = get_support_sum(train_x, train_y, lambdas, kern, support_ind, True)
    b = train_y[support_ind] - sum_for_support

    return lambdas, b


def get_support_sum(dataset_x, dataset_y, lambdas, kern, x, ind=False):
    summ = 0
    for i in range(len(dataset_x)):
        if lambdas[i] > EPS:
            summ += lambdas[i] * dataset_y[i] * (mem[i][x] if ind else kern(dataset_x[i], x))
    return summ


def get_ans(dataset_x, dataset_y, lambdas, w_0, kern, x):
    return np.sign(get_support_sum(dataset_x, dataset_y, lambdas, kern, x) + w_0)


def cv(dataset_x, dataset_y, kern, c):
    my_anses = []
    pack_size = 10
    ind = 0
    while ind < len(dataset_y):
        cur_pack_size = min(pack_size, len(dataset_y) - ind)
        to_delete = range(ind, ind + cur_pack_size)
        newdataset_x = np.delete(dataset_x, to_delete, 0)
        newdataset_y = np.delete(dataset_y, to_delete, 0)
        lambdas, w_0 = find_lambda(newdataset_x, newdataset_y, kern, c)
        for i in range(ind, ind + cur_pack_size):
            my_anses.append(get_ans(newdataset_x, newdataset_y, lambdas, w_0, kern, dataset_x[i]))
        ind += cur_pack_size
    return get_accuracy(my_anses, dataset_y)


def find_hyperparams():
    Cs = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    ds = [2, 3, 4, 5]
    shifts = [0]
    betas = [1, 2, 3, 4, 5]
    kernels = [] + \
              [Kernels.linear_fabric(shift) for shift in shifts] + \
              [Kernels.gauss_fabric(beta) for beta in betas] + \
              [cheety_kernel1, cheety_kernel2] + \
              [Kernels.poly_fabric(d, 1) for d in ds] + \
              [Kernels.poly_fabric(d) for d in ds] + \
              []

    dataset_names = ["geyser.csv", "chips.csv"]

    for name in dataset_names:
        dataset_x, dataset_y = read_data(name)
        max_params = [name, Cs[0], kernels[0]]
        for kern in kernels:
            max_ans = 0
            for C in Cs:
                # print("C={}, kern: {}".format(C, kern))
                ans = cv(dataset_x, dataset_y, kern, C)
                if ans > max_ans:
                    max_ans = ans
                    max_params = [name, C, str(kern)]
                    # print("max has updated", max_ans, max_params)

            print(max_params, "max_ans: ", max_ans)


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


def draw_fun_abstract(bounds, f, ax_xlabel="", ax_ylabel="", title=""):
    left, right, step_size = bounds
    step_count = int(np.floor((right - left) / step_size))
    # right = left + step_count * step_size

    plot_x = [left + i * step_size for i in range(step_count)]
    plot_y = [f(x) for x in plot_x]

    fig, ax = plt.subplots()
    ax.set_xlabel(ax_xlabel)
    ax.set_ylabel(ax_ylabel)
    plt.title(title)
    plt.plot(plot_x, plot_y)

    return plot_x, plot_y


def draw_one_set(dataset_name, paramss):
    dataset_x, dataset_y = read_data(dataset_name)

    for params in paramss:
        _draw(dataset_x, dataset_y, params)
        plt.title("Датасет: {}, accuracy при c-v = {}\nЯдро: {}".format(dataset_name, params[-1], params[1]))
        plt.show()


def draw():

    geyser_paramss = [
        # [0.5, cheety_kernel1, 0.8918],
        # [0.5, cheety_kernel2, 0.8918],
        [0.05, Kernels.poly_fabric(2, 1), 0.9009],
        [0.05, Kernels.poly_fabric(3, 1), 0.8783],
        [0.05, Kernels.poly_fabric(4, 1), 0.8648],
        [0.05, Kernels.poly_fabric(5, 1), 0.8828],
        [1.0, Kernels.gauss_fabric(1), 0.9054],
        [1.0, Kernels.gauss_fabric(2), 0.9054],
        [0.5, Kernels.gauss_fabric(3), 0.9054],
        [0.5, Kernels.gauss_fabric(4), 0.9054],
        [0.1, Kernels.gauss_fabric(5), 0.9009],
        [0.5, Kernels.linear_fabric(), 0.8648]
    ]
    draw_one_set("geyser.csv", geyser_paramss)

    chips_paramss = [
        [10.0, cheety_kernel1, 0.7457],
        [10.0, cheety_kernel2, 0.8135],
        [10.0, Kernels.poly_fabric(2, 1), 0.7542],
        [5.0, Kernels.poly_fabric(3, 1), 0.8220],
        [1.0, Kernels.poly_fabric(4, 1), 0.8220],
        [0.5, Kernels.poly_fabric(5, 1), 0.7203],
        [10.0, Kernels.gauss_fabric(1), 0.7966],
        [5.0, Kernels.gauss_fabric(2), 0.7627],
        [0.5, Kernels.gauss_fabric(3), 0.7542],
        [0.5, Kernels.gauss_fabric(4), 0.7627],
        [0.5, Kernels.gauss_fabric(5), 0.7966],
        [100.0, Kernels.linear_fabric(), 0.5084]
    ]
    draw_one_set("chips.csv", chips_paramss)


def _draw(dataset_x, dataset_y, hyperparams):
    squares_per_line = 200
    sqrrng = range(squares_per_line)
    sqr_len = 1.0 / squares_per_line
    rect_coord = [x * sqr_len + sqr_len / 2.0 for x in sqrrng]

    C = hyperparams[0]
    kern = hyperparams[1]
    lambdas, w_0 = find_lambda(dataset_x, dataset_y, kern, C)
    f = partial(get_ans, dataset_x, dataset_y, lambdas, w_0, kern)

    data = [[f([rect_coord[x], rect_coord[y]]) for y in sqrrng] for x in sqrrng]

    def get_rgb(x):
        return [0, 0, 0] if x == 1 else [255, 255, 255]

    rgb_data = [[get_rgb(data[x][y]) for x in sqrrng] for y in sqrrng]
    fig, ax = plt.subplots()
    # ax.imshow(data, interpolation='nearest', origin='lower', cmap=cmap, norm=norm,  extent=[0, 1, 0, 1])
    ax.imshow(rgb_data, interpolation='quadric', extent=[0, 1, 0, 1], origin='lower')

    draw_dataset(dataset_x, dataset_y)


def test():
    myans = [-1, -1, 1, 1, 1, -1, -1, -1, -1, -1]
    rigans = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
    print(get_accuracy(myans, rigans))


FILE_NAME = "chips.csv"
SEED = 31415
EPS = 1e-11
mem = []

#X, Y = read_data(FILE_NAME)

# test()

#draw()

# _draw(X, Y, [50, cheety_kernel1])
# plt.show()
# _draw(X, Y, [10, cheety_kernel2])
# plt.show()

# print(cv(X, Y, cheety_kernel1, 1))
# print(cv(X, Y, cheety_kernel2, 1))
# find_hyperparams()

