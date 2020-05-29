import numpy as np
import pandas as pd



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_der(x):
    return 1 - x * x


def sqrt(x):
    return np.math.sqrt(x)


def sin(x):
    return np.math.sin(x)


def calculate_MSE(targets, outputs):
    result = 0
    for i in range(len(targets)):
        result += (targets[i] - outputs[i]) ** 2
    return result

def calculate_results_table(number_of_classes, expected, actual, title):
    result_tab = np.zeros(shape=(number_of_classes, number_of_classes))

    for i in range(len(actual)):
        result_tab[expected[i]][np.argmax(actual[i])] += 1

    print(title)
    print(result_tab)


def parameters_as_string(hidden_nodes, learning_rate, epochs, learning_range):
    result = ''
    result += 'Hidden nodes= ' + str(hidden_nodes) + ' | ' \
            + 'Learning rate= ' + str(learning_rate) + ' | ' \
            + 'Epochs= ' + str(epochs) + '\n' \
            + 'Range: [' + str(learning_range[0]) \
            + '; ' + str(learning_range[1]) + ']'
    return result


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step
