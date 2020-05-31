import csv
from sklearn import datasets
import numpy as np
from mlxtend.plotting import plot_decision_regions

import rbf
import matplotlib.pyplot as plt
import os

filename = str("data/iris.csv")
iris = np.genfromtxt(filename, delimiter=',', dtype=['<f8', '<f8', '<f8', '<f8', 'U15'],
                     names=('sepal length', 'sepal width', 'petal length', 'petal width', 'label'))

# Create target Directory
path = "results/classification"
try:
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")


# division data to train_data and test_data
def data_division(data_training, train_data_percent):
    amount_of_data_types = len(data_training) / 3
    division = round(amount_of_data_types * train_data_percent)
    data_train = []
    data_test = []
    for i in range(0, division):
        data_train.append(data_training[i])
    for i in range(division, round(amount_of_data_types)):
        data_test.append(data_training[i])
    for i in range(round(amount_of_data_types), round(amount_of_data_types) + division):
        data_train.append(data_training[i])
    for i in range(round(amount_of_data_types) + division, round(amount_of_data_types) * 2):
        data_test.append(data_training[i])
    for i in range(round(amount_of_data_types) * 2, round(amount_of_data_types) * 2 + division):
        data_train.append(data_training[i])
    for i in range(round(amount_of_data_types) * 2 + division, len(data_train)):
        data_test.append(data_training[i])
    return data_train, data_test


# transform data from [1,1,1,1,Iris-setosa] to [1,1,1,1,0,0,1]
def transform_data(old_data, row_n, inputs_n, outputs_n, columns=None):
    if columns is None:
        columns = range(inputs_n)
    temp = np.zeros(shape=(len(old_data), len(columns)))
    new_data = np.ndarray(shape=(row_n, inputs_n + outputs_n))
    for i in range(len(old_data)):
        j = 0
        for col in columns:
            temp[i][j] = old_data[i][col]
            j += 1
        pom = temp[i]
        if old_data[i][-1] == 'Iris-setosa':
            new_data[i] = np.append(pom, [1, 0, 0])
        elif old_data[i][-1] == 'Iris-versicolor':
            new_data[i] = np.append(pom, [0, 1, 0])
        elif old_data[i][-1] == 'Iris-virginica':
            new_data[i] = np.append(pom, [0, 0, 1])
    return new_data


# Testing network, testing data, result data, number of inputs, correctness percentage
def classification_test(t_network, t_data_conv, t_data, inputs_n, correct_p):
    i = correct = 0
    for row in t_data_conv:
        result = ''
        pred = t_network.predict(row[:inputs_n])
        if pred[0] > correct_p:
            result = 'Iris-setosa'
        elif pred[1] > correct_p:
            result = 'Iris-versicolor'
        elif pred[2] > correct_p:
            result = 'Iris-virginica'
        if result == t_data[i][-1]:
            correct += 1
        i += 1
    return correct * 100 / len(t_data)


def change_to_iris_name(pred, correct_p):
    result = ''
    if pred[0] > correct_p:
        result = 'Iris-setosa'
    elif pred[1] > correct_p:
        result = 'Iris-versicolor'
    elif pred[2] > correct_p:
        result = 'Iris-virginica'
    else:
        print(pred, ' bad prediction')
    return result


def change_iris_name_to_int(data_train, data_test):
    x = []
    y = []
    for row in data_train:
        if row[-1] == 'Iris-setosa':
            x.append(1)
        elif row[-1] == 'Iris-versicolor':
            x.append(2)
        elif row[-1] == 'Iris-virginica':
            x.append(3)
    for row in data_test:
        if row[-1] == 'Iris-setosa':
            y.append(1)
        elif row[-1] == 'Iris-versicolor':
            y.append(2)
        elif row[-1] == 'Iris-virginica':
            y.append(3)
    return x, y

def accuracy_test(data_test, network):
    y_predict = network.predict(data_test[:, :-3])
    result = 0
    for i in range(len(data_test)):
        origin = change_to_iris_name(y_predict[i], 0.7)
        pred = change_to_iris_name(data_test[i, -3:], 0.7)
        if origin == pred:
            result += 1
        print(data_test[i, :], " : ", origin, " = ", pred)
    result = result / len(data_test) * 100
    print('Result: ', result, "%")
    return result


def classification_accuracy(train_data, test_data, columns):
    plt.title("Classification accuracy\n" +
              "(number of inputs " + str(len(columns)) + ", columns " + str(columns) + ")")
    plt.xlabel('number of radial neurons')
    plt.ylabel('accuracy')
    plt.grid(alpha=.4, linestyle='--')
    x = [0, 1, 11, 21, 31, 41]
    y = [0]
    for n_neurons in range(1, 42, 10):
        net = rbf.RBF(n_neurons)
        net.fit_two_stages(train_data[:, :-3], train_data[:, -3:], False)
        y.append(accuracy_test(test_data, net))
    plt.plot(x, y, label='train data', alpha=0.8)
    y = [0]
    for n_neurons in range(1, 42, 10):
        net = rbf.RBF(n_neurons)
        net.fit_two_stages(test_data[:, :-3], test_data[:, -3:], False)
        y.append(accuracy_test(test_data, net))
    plt.plot(x, y, label='test data', alpha=0.8)
    plt.legend()
    plt.savefig(
        'results/classification/classification_accuracy_' + str(len(columns)) + '_inputs_c' + str(columns) + '.png')
    plt.clf()


def classification_test_table(data_test, data_train):
    test_dataset = transform_data(data_test, len(data_test), 4, 3)
    train_dataset = transform_data(data_train, len(data_train), 4, 3)
    with open('results/classification/classification_effect_table.txt', 'w', newline='') as fileOut:
        writer = csv.writer(fileOut, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['hidden_neurons', 'train_accuracy', 'st_d_train', 'test_accuracy', 'st_d_test'])
        for i in range(1, 42, 5):
            acc_train = []
            acc_test = []
            for j in range(100):
                network = rbf.RBF(i)
                network.fit_two_stages(train_dataset[:, :-3], train_dataset[:, -3:], False)
                acc_train.append(accuracy_test(train_dataset, network))
                acc_test.append(accuracy_test(test_dataset, network))
            writer.writerow([i + 1, round(np.average(acc_train), 1), round(np.std(acc_train), 2),
                             round(np.average(acc_test), 1), round(np.std(acc_test), 2)])


def decision_boundaries():
    for i in range(3):
        for j in range(i + 1, 4):
            columns = [i, j]
            plt.title("Distribution of points for inputs number " + str(i + 1) + " and " + str(j + 1))
            plt.xlabel('input ' + str(i + 1))
            plt.ylabel('input ' + str(j + 1))
            plt.grid(alpha=.4, linestyle='--')
            iris = datasets.load_iris()
            X = iris.data[:, columns]
            y = iris.target
            network = rbf.RBF(31)
            network.fit_two_stages(X, y, False)
            # Plotting decision regions
            plot_decision_regions(X, y, clf=network, legend=2)
            plt.savefig('results/classification/2222distribution_input_' + str(i + 1) + '_and_' + str(j + 1))
            plt.clf()


data_train, data_test = data_division(iris, 0.8)

# Task 1

# Testing neural network with 1 Input
for i in range(4):
    print("Testing neural network with 1 Input, for data column " + str(i))
    training_data = transform_data(data_train, len(data_train), 1, 3, [i])
    test_data = transform_data(data_test, len(data_test), 1, 3, [i])
    classification_accuracy(training_data, test_data, [i])

# Testing neural network with 2 Inputs
for i in range(4):
    for j in range(i + 1, 4):
        print("Testing neural network with 2 Inputs, for data columns [" + str(i) + ", " + str(j) + "]")
        training_data = transform_data(data_train, len(data_train), 2, 3, [i, j])
        test_data = transform_data(data_test, len(data_test), 2, 3, [i, j])
        classification_accuracy(training_data, test_data, [i, j])

# Testing neural network with 3 Inputs
for i in range(4):
    for j in range(i + 1, 4):
        for k in range(j + 1, 4):
            print("Testing neural network with 3 Inputs, for data columns [" + str(i) + ", " + str(j) + ", " + str(
                k) + "]")
            training_data = transform_data(data_train, len(data_train), 3, 3, [i, j, k])
            test_data = transform_data(data_test, len(data_test), 3, 3, [i, j, k])
            classification_accuracy(training_data, test_data, [i, j, k])

# Testing neural network with 4 Inputs
print("Testing neural network with 4 Inputs, for data columns [0,1,2,3]")
training_data = transform_data(data_train, len(data_train), 4, 3)
test_data = transform_data(data_test, len(data_test), 4, 3)
classification_accuracy(training_data, test_data, [0, 1, 2, 3])

# Task 2

# classification_test_table(data_test, data_train)

# Task 3

# decision_boundaries()
