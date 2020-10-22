from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from functools import partial

np.set_printoptions(suppress=True)


def read_dataset(inp, start):
    return np.array(inp[start + 1:start + 1 + int(inp[start][0])])


def read_data(f):
    inp = []
    with open(f, 'r') as my_input:
        for line in my_input:
            inp.append([float(i) for i in line.rstrip('\n').split(' ')])

    train = read_dataset(inp, 1)
    test = read_dataset(inp, len(train) + 2)
    return train, test


def get_minmax(dataset):
    return [[dataset[:, i].min(), dataset[:, i].max()] for i in range(len(dataset[0]))]


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            num = row[i] - minmax[i][0]
            den = minmax[i][1] - minmax[i][0]
            row[i] = num / den if den != 0 else 0


def get_ans(w, x_i):
    return np.dot(w, x_i)


def get_y(dataset):
    dataset_y = dataset[:, -1].copy()
    for i in range(len(dataset)):
        dataset[i][-1] = 1
    return dataset_y


def do_train(train, max_iters, alpha, theta, k, b, shuffle=False, plot=False, ws=None, errs=None):
    if errs is None:
        errs = []
    if ws is None:
        ws = []
    np.random.seed(SEED)

    if shuffle:
        perm = np.random.permutation(len(train))
    else:
        perm = []

    n = len(train[0])
    a = 1.0 / (2 * n)
    w = [np.random.random() * 2 * a - a for _ in range(n)]
    ans = get_ans(w, train[0])
    err = (ans - dataset_train_y[0]) ** 2

    for step in range(1, max_iters + 1):
        mu = float(k / step + b)
        ind = perm[(step - 1) % len(train)] if shuffle else np.random.randint(len(train))
        ans = get_ans(w, train[ind])
        err = (1 - alpha) * err + alpha * (ans - dataset_train_y[ind]) ** 2
        der = 2 * (ans - dataset_train_y[ind])
        if der > MAX_DER:
            if SHOW_WARNINGS:
                print("step configuration mu={0}/step+{1} has divergent on step {2}.".format(k, b, step))
                print("will be returned current value of w")
            return 314159265358, w
        w = np.dot((1 - mu * theta), w) - np.dot(mu * der, train[ind])
        if plot:
            errs.append(err)
            ws.append(w)

    return err, w


def do_test(w, test, test_y):
    anses = [np.dot(w, x_i) for x_i in test]
    return sqrt(mean_squared_error(test_y, anses))


def tern_step(L, R, eps, curk, curb, f, k_or_b):
    if R - L > eps:
        p1 = L + (R - L) / 3
        p2 = R - (R - L) / 3
        ans1, _ = f(p1, curb) if k_or_b else f(curk, p1)
        ans2, _ = f(p2, curb) if k_or_b else f(curk, p2)
        if ans1 > ans2:
            L = p1
        else:
            R = p2
    return L, R


def do_train_plus(train, max_iters, alpha, theta):
    k_bounds = [0.0, 1.0, 0.01]
    b_bounds = [0.0, 1.0, 0.01]

    f = partial(do_train, train, max_iters, alpha, theta)
    kl = k_bounds[0]
    kr = k_bounds[1]
    keps = k_bounds[2]
    bl = b_bounds[0]
    br = b_bounds[1]
    beps = b_bounds[2]
    curk = (kr + kl) / 2
    curb = (br + bl) / 2
    while kr - kl > keps and br - bl > beps:
        kl, kr = tern_step(kl, kr, keps, curk, curb, f, True)
        curk = (kr + kl) / 2
        bl, br = tern_step(bl, br, beps, curk, curb, f, False)
        curb = (br + bl) / 2

    return curk, curb


def do_test_plus(train, test, max_iters, alpha, theta, shuffle=False):
    k, b = do_train_plus(train, max_iters, alpha, theta)
    _, w = do_train(train, max_iters, alpha, theta, k, b, shuffle)
    ans = do_test(w, test, dataset_test_y)
    return ans


def draw_fun_abstract(bounds, f):
    left, right, step_size = bounds
    step_count = int(np.floor((right - left) / step_size))
    # right = left + step_count * step_size

    plot_x = [left + i * step_size for i in range(step_count)]
    plot_y = [f(x) for x in plot_x]

    return plot_x, plot_y


def draw_fun_abstract_min(bounds, f):
    plot_x, plot_y = draw_fun_abstract(bounds, f)
    left, right, step_size = bounds
    argmin = round(np.argmin(plot_y) * step_size + left, 4)
    minimum = round(np.min(plot_y), 4)
    return plot_x, plot_y, argmin, minimum


def _draw_thetas(bounds, max_iters, alpha, dataset_name, shuffle=False):
    global dataset_train, dataset_test, dataset_train_y, dataset_test_y
    dataset_train, dataset_test, dataset_train_y, dataset_test_y = init("LR/" + dataset_name + ".txt")

    f = partial(do_test_plus, dataset_train, dataset_test, max_iters, alpha, shuffle=shuffle)
    plot_x, plot_y, argmin, minimum = draw_fun_abstract_min(bounds, f)

    fig, ax = plt.subplots()
    ax.set_xlabel('theta (параметр регуляризации)')
    ax.set_ylabel('Среднеквадратичная ошибка на тесте')
    plt.title("Датасет \"{}\", alpha = {}, {}\nMinimum: ({}, {})"
              .format(dataset_name, alpha, "shuffled" if shuffle else"", argmin, minimum))
    plt.plot(plot_x, plot_y)
    plt.show()


def _draw_alphas(bounds, max_iters, theta, shuffle=False):
    f = partial(do_test_plus, dataset_train, dataset_test, max_iters, theta=theta, shuffle=shuffle)
    plot_x, plot_y, argmin, minimum = draw_fun_abstract_min(bounds, f)

    fig, ax = plt.subplots()
    ax.set_xlabel('alpha (скорость забывания)')
    ax.set_ylabel('Среднеквадратичная ошибка на тесте')
    plt.title("theta = {}, {}\nMinimum: ({}, {})"
              .format(theta, "shuffled" if shuffle else"", argmin, minimum))
    plt.plot(plot_x, plot_y)
    plt.show()


def draw_alphas():
    bounds = [0, 1, 0.001]
    _draw_alphas(bounds, ITER_BOUND, 0.27, True)
    _draw_alphas(bounds, ITER_BOUND, 0.27)


def _draw_iters(alpha, theta, dataset_name):
    global dataset_train, dataset_test, dataset_train_y, dataset_test_y
    dataset_train, dataset_test, dataset_train_y, dataset_test_y = init("LR/" + str(dataset_name) + ".txt")

    plot_iter_x = [i for i in range(1, ITER_BOUND + 1)]
    plot_iter_y_errs = []
    ws = []
    k, b = do_train_plus(dataset_train, ITER_BOUND, alpha, theta)
    do_train(dataset_train, ITER_BOUND, alpha, theta, k, b, False, True, ws, plot_iter_y_errs)
    plot_iter_y_train_nrmse = [do_test(w_i, dataset_train, dataset_train_y) for w_i in ws]
    plot_iter_y_test_nrmse = [do_test(w_i, dataset_test, dataset_test_y) for w_i in ws]

    fig, ax = plt.subplots()
    ax.set_xlabel('Количество итераций')
    ax.set_ylabel('Среднеквадратическая ошибка/риск')
    plt.title("Датасет \"{}\", alpha = {}, theta={}\nМинимум: {}"
              .format(dataset_name, alpha, theta, round(min(plot_iter_y_test_nrmse), 4)))
    plt.plot(plot_iter_x, plot_iter_y_errs, '-r', label="Risk")
    plt.plot(plot_iter_x, plot_iter_y_test_nrmse, '-b', label="Test nrmse")
    plt.plot(plot_iter_x, plot_iter_y_train_nrmse, '-g', lw=1, label="Train nrmse")
    plt.legend()
    plt.show()


def draw_iters():
    alphas = [0.5]
    thetas = [0, 0.6, 1]
    for ds in range(1, 8):
        for theta in thetas:
            for alpha in alphas:
                _draw_iters(alpha, theta, ds)


def work_MHK(theta):
    F = dataset_train
    y = dataset_train_y
    V, D_vec, Ut = np.linalg.svd(F, full_matrices=False)
    D = np.diag(D_vec)
    D_rev_reg = np.diag([1 / (i * i + theta) for i in D_vec])
    F_pinv = Ut.T.dot(D_rev_reg).dot(D).dot(V.T)
    w = np.dot(F_pinv, y)
    return do_test(w, dataset_test, dataset_test_y)


def _draw_thetas_MHK(dataset_name, bounds):
    global dataset_train, dataset_test, dataset_train_y, dataset_test_y
    dataset_train, dataset_test, dataset_train_y, dataset_test_y = init("LR/" + str(dataset_name) + ".txt")

    plot_x, plot_y, argmin, minimum = draw_fun_abstract_min(bounds, work_MHK)

    fig, ax = plt.subplots()
    ax.set_xlabel('theta (параметр регуляризации)')
    ax.set_ylabel('Среднеквадратичная ошибка на тесте')
    plt.title("Датасет \"{}\", minimum: ({}, {})"
              .format(dataset_name, argmin, minimum))
    plt.plot(plot_x, plot_y)
    plt.show()


def draw_thetas_MHK(bounds):
    # draw_thetas_MHK([0, 0.002, 0.0001])
    for ds in range(1, 8):
        _draw_thetas_MHK(ds, bounds)


def init(name):
    train, test = read_data(name)
    minmax = get_minmax(train)
    normalize(train, minmax)
    normalize(test, minmax)
    train_y = get_y(train)
    test_y = get_y(test)
    return train, test, train_y, test_y


def draw_thetas():
    _draw_thetas([0, 100, 1], ITER_BOUND, 0.5, str(3))

    bounds = [[0, 1, 0.01], [0, 10, 0.1], [0, 100, 1], [0, 1000, 10]]
    for ds in range(1, 8):
        for bound in bounds:
            _draw_thetas(bound, ITER_BOUND, 0.5, str(ds))


FILENAME = 'LR/1.txt'
ITER_BOUND = 2000
MAX_DER = 100000
SHOW_WARNINGS = False
SEED = 31415


dataset_train, dataset_test, dataset_train_y, dataset_test_y = init(FILENAME)

draw_thetas_MHK([0, 0.02, 0.0001])

# draw_thetas()
# draw_alphas()
# draw_iters()
