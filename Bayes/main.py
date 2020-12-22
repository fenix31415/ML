import os
from collections import defaultdict

import numpy as np
from numpy import log, math
from enum import Enum
from matplotlib import pyplot as plt

DIR_PREFIX = "./messages/part"


class EmailType(Enum):
    SPAM = 0
    LEGIT = 1


def read_data(n=0):
    folds = []
    folds_count = 10
    for k in range(1, folds_count + 1):
        cur_fold = []
        dir_name = DIR_PREFIX + str(k) + "/"
        files = os.listdir(dir_name)
        for file in files:
            with open(dir_name + file, 'r') as f:
                is_legit = EmailType.LEGIT if file.find("legit") != -1 else EmailType.SPAM
                lines = f.readlines()
                header = list(map(lambda _word: "_" + _word, lines[0].lstrip('Subject: ').split()))
                text = lines[2].split() + header
                new_text = []
                for ind in range(len(text) - n):
                    word = text[ind]
                    for i in range(n):
                        word = word + "_" + text[ind + i + 1]
                    new_text.append(word)
                cur_fold.append([is_legit, new_text])
        folds.append(cur_fold)
    return folds


def train(is_old, train_dataset, alpha):
    aprior_prop = np.zeros(2)
    classes_words_freqs = defaultdict(lambda: 0.0)
    data = [[defaultdict(lambda: 0.0) for _ in range(2)] for _ in range(2)]
    words_set = set()
    default_data = [0.0 for _ in EmailType]

    all_words = 0.0
    for is_legit, text in train_dataset:
        word_count = len(text)
        all_words += word_count
        aprior_prop[is_legit.value] += word_count
    aprior_prop[0] /= all_words
    aprior_prop[1] /= all_words

    if is_old:
        words_count = defaultdict(lambda: 0.0)
        for is_legit, text in train_dataset:
            for word in text:
                classes_words_freqs[is_legit.value, word] += 1
                words_count[word] += 1
        for c, word in classes_words_freqs:
            classes_words_freqs[c, word] /= words_count[word]
    else:
        count = [[defaultdict(lambda: 0.0) for _ in range(2)] for _ in range(2)]
        size_of_classes = [0 for _ in EmailType]
        for is_legit, text in train_dataset:
            words_set = words_set.union(set(text))
        for is_legit, text in train_dataset:
            size_of_classes[is_legit.value] += 1
            for word in set(text):
                count[is_legit.value][0][word] += 1
        for word in words_set:
            for is_legit_enum in EmailType:
                count[is_legit_enum.value][1][word] = size_of_classes[is_legit_enum.value]
        for word in words_set:
            for _in in range(2):
                for is_legit_num in EmailType:
                    is_legit = is_legit_num.value
                    data[is_legit][_in][word] = (count[is_legit][_in][word] + alpha) / \
                                                (count[is_legit][0][word] + count[is_legit][1][word] + 2 * alpha)
        for is_legit_enum in EmailType:
            for word in words_set:
                default_data[is_legit_enum.value] += log(data[is_legit_enum.value][1][word] + 10 ** (-10))

    return aprior_prop, classes_words_freqs if is_old else [words_set, default_data, data]


def get_prop_product(class_id, is_old, train_data, email):
    ans = 0.0
    eps = 10 ** (-10)
    if not is_old:
        words_set, default_data, data = train_data
        ans = default_data[class_id]
        # for word in words_set:
        #    ans += log(data[class_id][0 if word in email else 1][word] + eps)
        for word in set(email):
            ans += log(data[class_id][0][word] + eps)
            ans -= log(data[class_id][1][word] + eps)
    else:
        for word in email:
            ans += log(train_data[class_id, word] + eps)
    return ans


def get_class(lambdas, aprior_prop, is_old, train_data, email):
    value = [0.0, 0.0]
    for class_id_emun in EmailType:
        class_id = class_id_emun.value
        lambda_val = (log(lambdas[class_id]) if class_id == EmailType.SPAM.value else lambdas[class_id])
        value[class_id] = log(aprior_prop[class_id]) + lambda_val
        value[class_id] += get_prop_product(class_id, is_old, train_data, email)
    spam_value = value[EmailType.SPAM.value]
    legit_value = value[EmailType.LEGIT.value]
    ans = 1 if legit_value >= spam_value else 0
    return ans, legit_value - spam_value


def cv(folds, lambdas, is_old, alpha, get_values=False):
    classes_rng = range(2)
    matrix = [[0 for _ in classes_rng] for _ in classes_rng]
    values = []
    for ind in range(10):
        test_dataset = folds[ind]
        train_dataset = []
        for i in range(10):
            if i != ind:
                for q in folds[i]:
                    train_dataset.append(q)
        aprior_prop, train_data = train(is_old, train_dataset, alpha)
        for rigans, email in test_dataset:
            myans, value = get_class(lambdas, aprior_prop, is_old, train_data, email)
            values.append([value, rigans])
            matrix[myans][rigans.value] += 1
    if get_values:
        values.sort(key=lambda v: v[0])

    ans = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0])
    return ans if (not get_values) else (ans, values)


def find_hyperparams():
    lambdass = [[1, i] for i in range(1, 11)]
    olds = [True, False]
    alphas = [0.1, 0.01, 1]
    ns = [1, 2, 3]

    max_ans = 0
    max_params = []
    for n in ns:
        folds = read_data(n)
        for is_old in olds:
            for alpha in (alphas if not is_old else [0]):
                for lambdas in lambdass:
                    ans = cv(folds, lambdas, is_old, alpha)
                    print([n, alpha, lambdas, is_old], ans)
                    if ans > max_ans:
                        max_ans = ans
                        max_params = [n, alpha, lambdas, is_old]
                        print("max has updated", max_ans, max_params)

    print(max_params, "max_ans: ", max_ans)


def draw_rock(paramss):
    xss = []
    yss = []
    for n, alpha, lambdas, old in paramss:
        _, values = cv(read_data(n), lambdas, old, alpha, True)
        x = y = 0.0
        xs = []
        ys = []
        for value, ans in values:
            if ans == EmailType.LEGIT:
                x += 1.0
            else:
                y += 1.0
            xs.append(x)
            ys.append(y)

        for i in range(len(xs)):
            xs[i] /= x
        for i in range(len(ys)):
            ys[i] /= y

        xss.append(xs)
        yss.append(ys)

    boundss = [[1, 0], [0.5, 0.5], [0.4, 0.6], [0.2, 0.8]]
    for x_bound, y_bound in boundss:
        for ind in range(len(paramss)):
            n, alpha, lambdas, old = paramss[ind]
            xs = xss[ind]
            ys = yss[ind]
            label = "lambdas = {}, old: {}, alpha = {}, n-gram = {}".format(lambdas, old, alpha, n)
            plt.plot(xs, ys, label=label)
            plt.legend(loc='lower right')
        plt.xlim(0, x_bound)
        plt.ylim(y_bound, 1)
        plt.show()


def draw():
    paramss = [
        [1, 0, [1, 9], True],
        [1, 0.1, [1, 1], False],
        [1, 0.01, [1, 1], False],
        [2, 0, [1, 7], True],
        [2, 0.01, [1, 3], False],
        [2, 0.1, [1, 1], False]
    ]
    draw_rock(paramss)


def find_nospam_lambda(params):
    n, alpha, is_old = params
    _, values = cv(read_data(n), [1, 1], is_old, alpha, True)
    legit_ind = 0
    while values[legit_ind][1] == EmailType.SPAM:
        legit_ind += 1
    legit_ind -= 1
    value_legit = values[legit_ind][0]
    value_spam = values[legit_ind - 1][0]
    c = (value_legit + value_spam) / 2.0
    # print(n, alpha, lambdas, is_old, ": lambda found: exp({}) = {}".format(-c, exp(-c)))  # exp(494.64) = 6.62
    return -c


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


def draw_nospam_(paramss):
    for params in paramss:
        lambda_upper = find_nospam_lambda(params)
        if math.isnan(lambda_upper) or math.isinf(lambda_upper):
            print("is to big", params)
            continue
        start_lambda = 1
        end_lambda = lambda_upper + 1
        step = lambda_upper / 10.0
        folds = read_data(params[0])

        def f(_x):
            ans = cv(folds, [1, _x], params[2], params[1], False)
            return ans

        # cv(folds, lambdas, is_old, alpha, get_values=False):
        x, y = draw_fun_abstract([start_lambda, end_lambda, step], f, title="lambdas")
        plt.plot(x, y)
        plt.title("Nospam lambda={}, n={}, alpha={}, is_old={}".format(lambda_upper, params[0], params[1], params[2]))
        plt.show()


def draw_nospam():
    paramss = [
        [1, 0, True],
        [2, 0, True],
        [1, 0.1, False],
        [1, 0.01, False],
        [2, 0.01, False],
        [2, 0.1, False]
    ]
    draw_nospam_(paramss)


# find_hyperparams()
# draw()
draw_nospam()
