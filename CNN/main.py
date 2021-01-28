from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend
import matplotlib.pyplot as plt

CLASSES_COUNT = 10


def read_dataset():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    return (train_x, train_y), (test_x, test_y)


def test_all(train_x, train_y, test_x, test_y):
    maxacc = 0
    maxmodel = None
    maxparams = None
    for params in get_params():
        ls = []
        for param in params:
            f, n, fun, b = param
            ls.append(layers.Conv2D(f, n, activation=fun, input_shape=(28, 28, 1), use_bias=b))
            ls.append(layers.MaxPooling2D((2, 2)))
        ls.append(layers.Flatten())
        ls.append(layers.Dense(100, activation='relu'))
        ls.append(layers.Dense(CLASSES_COUNT, activation='softmax'))

        model = keras.Sequential(ls)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_x, train_y, epochs=10, verbose=2)
        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
        if test_acc > maxacc:
            maxmodel = model
            maxparams = params
            maxacc = test_acc
    print(maxacc, maxparams)
    do_matrix(maxmodel, test_x, test_y)


def do_matrix(model, test_x, test_y):
    anses = model.predict(test_x)
    rng = range(CLASSES_COUNT)
    matrix = [[0 for _ in rng] for _ in rng]
    data = [(-1, 0.0) for _ in range(CLASSES_COUNT * CLASSES_COUNT)]

    for (i, ps) in enumerate(anses):
        ans = keras.backend.argmax(ps)
        if ps[ans] > data[test_y[i] * CLASSES_COUNT + ans][1]:
            data[test_y[i] * CLASSES_COUNT + ans] = (i, ps[ans])
        matrix[test_y[i]][ans] += 1

    for i in range(10):
        print(matrix[i])
    draw(data, test_x)


def draw(data, test_x):
    blank_image = [[255 for _ in range(28)] for _ in range(28)]
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=0.8, hspace=0.1)
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if data[i][0] == -1:
            plt.imshow(blank_image, cmap=plt.cm.binary)
            continue
        plt.imshow(test_x[data[i][0]], cmap=plt.cm.binary)
        plt.xlabel(i)
    plt.show()


def get_params():
    for n in [3, 5]:
        for bias in [True, False]:
            for fun in ["relu", "sigmoid", "tanh"]:
                yield [[10, n, fun, bias], [20, n, fun, bias]]
                yield [[32, n, fun, bias], [64, n, fun, bias]]
                yield [[10, n, fun, bias], [100, n, fun, bias]]


def test(train_x, train_y, test_x, test_y):
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=5, activation='relu', input_shape=(28, 28, 1), use_bias=True),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=5, activation='relu', use_bias=True),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=10, verbose=0)
    model.evaluate(test_x, test_y, verbose=2)
    do_matrix(model, test_x, test_y)


def main():
    (train_x, train_y), (test_x, test_y) = read_dataset()
    test(train_x, train_y, test_x, test_y)
    # test_all(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
