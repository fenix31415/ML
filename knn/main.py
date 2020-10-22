from functools import partial
from math import exp
from math import pi

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0]) - 1):
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def label_enc():
    keys = [1, 2, 3]
    values = [0, 1, 2]
    return dict(zip(keys, values))


def get_neighbors(train, x, N, p):
    distances = list()
    for x_i in train.values:
        dist = p(x, x_i)
        distances.append((x_i, dist))
    distances.sort(key=lambda tup: tup[1])

    neighbors = list()
    for i in range(N):
        neighbors.append(distances[i][0])
    return neighbors


def get_ans_1(train, x, n, w, p):
    neighbors = get_neighbors(train, x, n, p)
    num = 0
    den = 0
    for x_i in neighbors:
        w_i = w(x_i, x)
        num += inds[x_i[-1]] * w_i
        den += w_i
    return round(num / den) if den != 0 else inds[neighbors[0][-1]]


def get_ans_2(train, x, n, w, p):
    neighbors = get_neighbors(train, x, n, p)
    vec = [0.0 for i in range(classes)]
    for i in range(classes):
        num = 0
        den = 0
        for x_i in neighbors:
            w_i = w(x_i, x)
            num += hot_inds[inds[x_i[-1]]][i] * w_i
            den += w_i
        vec[i] = num / den if den != 0 else 0
    return np.argmax(vec)


def cv(dataset, n, w, p, type):
    if n == 0:
        return 0
    matrix = [[0 for x in range(classes)] for y in range(classes)]
    for i in range(len(dataset.values)):
        myans = get_ans_1(dataset.drop(i), dataset.values[i], n, w, p) if type == 0 else get_ans_2(dataset.drop(i),
                                                                                                   dataset.values[i], n,
                                                                                                   w, p)
        rigans = inds[dataset.values[i][-1]]
        matrix[myans][rigans] += 1
    sum_colls = [sum(a) for a in zip(*matrix)]
    beta = 1
    prec = [0.0 for x in range(classes)]
    recall = [0.0 for x in range(classes)]
    f_scope = [0.0 for x in range(classes)]
    for c in range(classes):
        sumrow = sum(matrix[c])
        prec[c] = matrix[c][c] / sumrow if sumrow != 0 else 1
        sumcoll = sum_colls[c]
        recall[c] = matrix[c][c] / sumcoll if sumcoll != 0 else 1
        f_scope[c] = (1 + beta) * (prec[c] * recall[c]) / (beta * prec[c] + recall[c])
    ans = sum(f_scope) / len(f_scope)
    return ans


def kernel_rec(x):
    return 0.5 * (1 >= x >= -1)


def kernel_tri(x):
    return (1 - abs(x)) * (1 >= x >= -1)


def kernel_qquad(x):
    return 15 / 16 * (1 - x * x) ** 2 * (1 >= x >= -1)


def kernel_parab(x):
    return 0.75 * (1 - x ** 2) * (1 >= x >= -1)


def kernel_gauss(x):
    return (2 * pi) ** (-1 / 2) * exp(-x * x / 2)


kernels = [kernel_rec, kernel_tri, kernel_qquad, kernel_parab, kernel_gauss]
kernels = [kernel_rec]


def dist_mink(r, a, b):
    ans = 0.0
    for i in range(len(a) - 1):
        ans += abs(a[i] - b[i]) ** r
    return ans ** (1 / r)


def find_params(dataset):
    r_values = [0.4, 0.5, 0.6, 1, 1.5, 2, 3, 4, 5, 10, 100]
    h_range = [0.5, 1, 1.5, 2, 3, 5, 10, 10.5, 11, 11.5, 12]
    n_range = range(1, 20)

    max_ans = 0
    max_params = [r_values[0], h_range[0], 0, 1, 0]

    for r in r_values:
        p = partial(dist_mink, r)
        for h in h_range:
            print(r, h, max_ans, max_params)
            for kernel_i in range(len(kernels)):
                def w(x_i, x):
                    return kernels[kernel_i](dist_mink(r, x_i, x) / h)

                for n in n_range:
                    for ans_type in range(2):
                        ans = cv(dataset, n, w, p, ans_type)
                        if ans > max_ans:
                            max_ans = ans
                            max_params = [r, h, kernel_i, n, ans_type]
                            print("max has apdated", max_ans, max_params)

    print(max_params, "max_ans: ", max_ans)
    return max_params


def draw_N(dataset, r, h, kernel, n_range, ans_type):
    p = partial(dist_mink, r)

    def w(x_i, x):
        return kernel(p(x_i, x) / h)

    x = [i for i in n_range]
    y = [cv(dataset, i, w, p, ans_type) for i in n_range]
    plt.plot(x, y)
    plt.show()


def draw_H(dataset, r, h_range, kernel, n, ans_type):
    p = partial(dist_mink, r)
    x = [0.5 * i for i in h_range]
    y = [0.0 for i in h_range]
    for i in h_range:
        h_i = 0.5 * i

        def w(x_i, x):
            return kernel(p(x_i, x) / h_i)

        y[i - 1] = cv(dataset, n, w, p, ans_type)
    plt.plot(x, y)
    plt.show()


def draw(dataset, r, h, kernel, n, ans_type):
    draw_H(dataset, r, range(1, 100), kernel, n, ans_type)
    draw_N(dataset, r, h, kernel, range(1, 50), ans_type)


def draw_kernels():
    x = [0.01 * i for i in range(-10, 110)]
    for kernel in kernels:
        y = [kernel(x_i) for x_i in x]
        plt.plot(x, y)
    plt.show()


classes = 3

filename = 'seeds.csv'
dataset = pd.read_csv(filename)
inds = label_enc()
hot_inds = [[1 if i == j else 0 for i in range(classes)] for j in range(classes)]
min_max = minmax(dataset.values)
normalize(dataset.values, min_max)
print(dataset.values[0])
# max_params = find_params(dataset)
# draw(dataset, max_params[0], max_params[1], kernels[max_params[2]], max_params[3], max_params[4])
draw(dataset, 0.5, 10.5, kernels[0], 9, 1)
