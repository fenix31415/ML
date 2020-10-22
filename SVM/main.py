import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt


def read_data():
    dataframe = pd.read_csv(FILE_NAME)
    dataset_y = np.array([1 if dataframe.to_numpy()[:, -1][i] == "P" else -1 for i in range(len(dataframe.to_numpy()))])
    dataframe.drop(columns=['class'], inplace=True)
    min_max = get_minmax(dataframe.to_numpy())
    dataset_x = normalize(dataframe.to_numpy(), min_max)
    return dataset_x, dataset_y


def get_minmax(data):
    return [[data[:, i].min(), data[:, i].max()] for i in range(len(data[0]) - 1)]


def normalize(data, minmax):
    for row in data:
        for i in range(len(row) - 1):
            num = row[i] - minmax[i][0]
            den = minmax[i][1] - minmax[i][0]
            row[i] = num / den if den != 0 else 0
    return data


def linear_kernel_fabric(shift=0):
    def prod(a, b):
        return np.asarray(a).dot(np.asarray(b)) + shift
    return prod


def poly_kernel_fabric(d):
    def prod(a, b):
        return (np.asarray(a).dot(np.asarray(b).T)) ** d
    return prod


def gauss_kernel_fabric(beta):
    def prod(a, b):
        return math.exp(-beta * np.linalg.norm(a - b) ** 2)
    return prod


def get_accuracy(my_anses, anses):
    return (np.asarray(my_anses) - np.asarray(anses)).count(0)


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


def fun(lambdas, y, k):
    ans = 0
    for i in range(len(lambdas)):
        ans -= lambdas[i]
        for j in range(len(lambdas)):
            ans += lambdas[i] * lambdas[j] * y[i] * y[j] * k(i, j)
    return ans


def find_lambda(train_y, kern, c):
    max_iters = 500

    size = len(train_y)
    lambdas = [c / 2.0] * size
    #lambdas = np.zeros(size)
    bounded = [False] * size

    mu = c
    koef = 0.9
    step_number = 0

    while (step_number < max_iters) and mu > 1e-15:
        step_number += 1
        delta = np.zeros(size)
        for k in range(size):
            summ = 0
            for i in range(size):
                summ += lambdas[i] * train_y[i] * train_y[k] * kern(k, i)
            if abs(lambdas[k] - c) > EPS:
                delta[k] = mu * (summ - 1)
            else:
                if not bounded[k]:
                    print(step_number, "bound", k)
                    bounded[k] = True
                    #mu = c
                delta[k] = 0
                lambdas[k] = c
        #delta = pr(delta, train_y)
        #newlambdas = lambdas - delta
        if step_number == 247:
            print()
        newlambdas = fix_pr(lambdas - delta, train_y, c)
        if is_in_cube(c, newlambdas):
            lambdas = newlambdas
            #print(step_number,newlambdas, mu, delta, abs(newlambdas[1]-10))
        else:
            mu *= koef
            #print(step_number,"rejected newlambda = {}, mu = {}"
            #      .format(newlambdas, mu))
    print(lambdas)
    return lambdas


def get_w(x, y, lambdas, ker):
    ans = np.zeros(len(x[0]))
    for i in range(len(x)):
        ans = ans + (lambdas[i] * y[i]) * x[i]
    ind = 0
    while abs(lambdas[ind]) < EPS:
        ind += 1
    w_0 = ans.dot(x[ind]) - y[ind]
    return ans, w_0


def get_ans(lambdas):
    sum = 0
    #for i in range(len())


def cv(dataset_x, dataset_y, kern, c):
    my_anses = []
    for i in range(len(dataset_y)):
        lambdas = find_lambda(dataset_y.pop(i), kern, c)
        w, w_0 = get_w(dataset_x, dataset_y, lambdas)
        my_anses.append(np.sign())
    return get_accuracy(my_anses, dataset_y)


FILE_NAME = "chips.csv"
SEED = 31415
EPS = 1e-11

X, Y = read_data()


def _f(k, i):
    return linear_kernel_fabric()(X[k], X[i])
    #return 1


lam = find_lambda(Y, _f, 10)
print(fun(lam, Y, _f))
print(fun(np.zeros(len(Y)), Y, _f))
print(fun([5] * len(Y), Y, _f))
print(fun([10] * len(Y), Y, _f))

