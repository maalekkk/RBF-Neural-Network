import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import rbf

# Create target directory
path = "results/approximation"
try:
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")

# Load TXT
filename = str("data/approx_1.csv")
with open(filename, 'r') as inFile:
    train_data = np.loadtxt(inFile, delimiter=" ")

filename = str("data/approx_test.csv")
with open(filename, 'r') as inFile2:
    test_data = np.loadtxt(inFile2, delimiter=" ")

train_data.sort(0)
test_data.sort(0)

train_x = train_data[:, 0, np.newaxis]
train_y = train_data[:, 1, np.newaxis]
test_x = test_data[:, 0, np.newaxis]
test_y = test_data[:, 1, np.newaxis]


def _plot_5_predict(num, step, train_x, train_y, test_x, backward=False,
                    coefficient=1.0, l_rate=0.01, b_rate=0.1, epochs=10000):
    for i in range(1, num, step):
        network = rbf.RBF(i)
        network.fit_two_stages(train_x, train_y, backward, coefficient, epochs, l_rate, b_rate)
        lab = str(i) + ' neuron(s)'
        plt.plot(test_x, network.predict(test_x), label=lab, linewidth=2, alpha=0.7)


def _plot_radial_functions(hidden_neurons, train_x, train_y, test_x, backward=True,
                           coefficient=1.0, l_rate=0.2, b_rate=0.2, epochs=1000):
    network = rbf.RBF(hidden_neurons)
    network.fit_two_stages(train_x, train_y, backward, coefficient, epochs, l_rate, b_rate)
    plt.plot(test_x, network.predict(test_x), label='network', linewidth=3)
    G = network.calculate_matrix(test_x, coefficient)
    for i in range(hidden_neurons):
        for j in range(len(test_x)):
            G[j][i] *= network.weights[i]
        lab = str(i) + 'neuron'
        plt.plot(test_x, G[:, i], label=lab, linewidth=2, alpha=0.7)


def mse_avg_table(train_x, train_y, test_x, test_y, file_name):
    with open(file_name, 'w', newline='') as fileOut:
        writer = csv.writer(fileOut, delimiter=' ',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['hidden_neuron', 'avg_mse_train', 'st_d_train', 'avg_mse_test', 'st_d_test'])
        for i in range(1, 42, 5):  # THIS RUNS FOR A LONG TIME!!! BECAUSE ITS 100 times training the neural network
            errors_train = []
            errors_test = []
            for j in range(100):
                print(str(j), '-ta iteracja')
                network = rbf.RBF(i)
                errors_train.append(network.fit_two_stages(train_x, train_y, True))
                errors_test.append(network.fit_two_stages(test_x, test_y, True))
            print(str(i) + ' neuron, errors train: ', errors_train)
            print(str(i) + ' neuron, errors test: ', errors_test)
            writer.writerow([j, round(np.average(errors_train), 4), round(np.std(errors_train), 4),
                             round(np.average(errors_test), 4), round(np.std(errors_test), 4)])


def task1(backward, coefficient=1.0):
    plt.title("Approximation result" + "\nbeta coefficient: " + str(coefficient))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(alpha=.4, linestyle='--')
    plt.scatter(test_data[:, 0], test_data[:, 1], c='blue', label="Test data", s=10, alpha=0.7)
    plt.scatter(train_data[:, 0], train_data[:, 1], c='k', label="Train data", marker='*', alpha=0.2)
    _plot_5_predict(42, 10, train_x, train_y, test_x, backward, coefficient)
    plt.legend()
    if backward:
        s = 'backward'
    else:
        s = 'pinv'
    plt.savefig('results/approximation/approx_task1_' + str(coefficient) + s + '.png')
    plt.clf()


def task2(hidden_neurons, coefficient=1.0):
    plt.title("Approximation radial functions" + "\nbeta coefficient: " + str(coefficient))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(alpha=.4, linestyle='--')
    plt.scatter(test_data[:, 0], test_data[:, 1], c='blue', label="Test data", s=10, alpha=0.7)
    plt.scatter(train_data[:, 0], train_data[:, 1], c='k', label="Train data", marker='*', alpha=0.2)
    _plot_radial_functions(hidden_neurons, train_x, train_y, np.linspace(-2, 3, 1000), True, coefficient)
    plt.legend(loc=2)
    plt.savefig('results/approximation/approx_task1_radial' + str(coefficient) + '.png')
    plt.clf()


def task4(hidden_neurons, train_x, train_y, test_x):
    time_jump = 5000
    color = ['brown', 'darkslategray', 'b', 'r', 'c', 'm', 'lime', 'g']
    plt.title("Approximation throughout epochs")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(alpha=.4, linestyle='--')
    net = rbf.RBF(hidden_neurons)
    net.fit_two_stages(train_x, train_y, True, 1, 1)
    plt.plot(test_x, net.predict(test_x), c=color[0], label=str(1) + " epoch")
    for j in range(1, 6):
        net.fit_two_stages(train_x, train_y, True, 1, time_jump)
        plt.plot(test_x, net.predict(test_x), c=color[j], label=str(j * time_jump) + " epoch", linewidth=1)
    plt.scatter(train_data[:, 0], train_data[:, 1], c='k', label="Train data", marker='*', alpha=0.2)
    plt.legend(loc=2)
    plt.savefig('results/approximation/approximation_throughout_epochs_data_1.png')
    plt.clf()


# # Task 1
# task1(True, 1)
# task1(True, 0.25)
# task1(True, 5)

# Task 2
# task2(11, 1)
# task2(11, 0.25)
# task2(11, 5)

# Task 3
# mse_avg_table(train_x, train_y, test_x, test_y, 'results/approximation/mse_table.txt')

# Task 4
# task4(21, train_x, train_y, test_x)
