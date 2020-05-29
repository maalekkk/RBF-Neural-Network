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
filename = str("data/approx_train_1.txt")
with open(filename, 'r') as inFile:
    train_data = np.loadtxt(inFile, delimiter=" ")

filename = str("data/approx_test.txt")
with open(filename, 'r') as inFile2:
    test_data = np.loadtxt(inFile2, delimiter=" ")

x = np.arange(-4, 4, 0.01)
color = ['brown', 'darkslategray', 'b', 'r', 'c', 'm', 'lime', 'g']


def predict(network, dataset):
    y = list()
    for row in dataset:
        y.append(network.prediction([row]))
    return y


def plott(num, step, data):
    n_epoch = 2000
    error_hist = list()
    epoch = list()
    color = ['darkslategray', 'b', 'r', 'c', 'm', 'lime', 'g']
    tmp = 0
    for i in range(1, num, step):
        network = rbf.RBF(i)
        error, it = network.fit_two_stages(data[:, 0], data[:, 1], True)
        lab = str(i) + ' neuron(s)'
        temp = list()
        for r in x:
            temp.append(network.predict([r]))
        plt.plot(x, temp, c=color[tmp], label=lab, linewidth=2)
        error_hist.append(error)
        epoch.append(it)
        tmp += 1


def error(steps, neurons, data, test_data, n_of_epoch):
    color = ['r', 'c', 'm', 'darkslategray', 'mediumpurple']
    # train data error
    for i in range(steps):
        network = rbf.RBF(neurons[i])
        error, it = network.fit_two_stages(data, n_of_epoch, epsilon)
        lab = str(neurons[i]) + ' (train)'
        plt.plot(range(it), error, label=lab, c=color[i], linewidth=1)
    # test data error
    for i in range(steps):
        network = rbf.RBF(neurons[i])
        error, it = network.fit_two_stages(test_data[:, 0], test_data[:, 1], True)
        lab = str(neurons[i]) + ' (test)'
        plt.plot(range(it), error, label=lab, linewidth=1)


def mse_avg_table(epoch, epsilon, lr, momentum, train_data, test_data, file_name):
    with open(file_name, 'w', newline='') as fileOut:
        writer = csv.writer(fileOut, delimiter=' ',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['hidden_neuron', 'avg_mse_train', 'st_d_train', 'avg_mse_test', 'st_d_test'])
        for i in range(20):  # THIS RUNS FOR A LONG TIME!!! BECAUSE ITS 100 times training the neural network
            errors_train = []
            errors_test = []
            for j in range(5):
                network = rbf.RBF(i + 1)
                errors_train.append(network.fit_two_stages(train_data[:, 0], train_data[:, 1], True))
                errors_test.append(network.fit_two_stages(test_data[:, 0], test_data[:, 1], True)[0][-1])
            writer.writerow([i + 1, round(np.average(errors_train), 4), round(np.std(errors_train), 4),
                             round(np.average(errors_test), 4), round(np.std(errors_test), 4)])


l_rate = 0.01
m_rate = 0.005
# Training data 1
plt.title("Approximation result for data 1" + '\nlearning rate = ' + str(l_rate) + ', momentum = ' + str(m_rate))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=.4, linestyle='--')
plt.scatter(test_data[:, 0], test_data[:, 1], c='lime', label="Test data", s=1)
plt.scatter(train_data[:, 0], train_data[:, 1], c='k', label="Train data", marker='*')
plott(20, 4, train_data)
plt.legend()
plt.savefig('results/approximation/approximation_result_data_1.png')
plt.clf()


l_rate = 0.002
m_rate = 0.00
epsilon = 0.01
n_of_epoch = 5000
# Error for 1, 5, 19 neurons in hidden layer, data1
neurons = [1, 5, 19]
steps = 3
plt.title("Error - first training set" + '\nlearning rate = ' + str(l_rate) + ', momentum = ' + str(m_rate))
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.ylim(0.0, 1.2)
plt.grid(alpha=.4, linestyle='--')
error(steps, neurons, train_data, test_data, n_of_epoch)
plt.legend()
plt.savefig('results/approximation/approximation_error_1.png')
plt.clf()


epoch = 3000
epsilon = 0.01
# # MSE error and standard deviation table
# mse_avg_table(epoch, epsilon, l_rate, m_rate, train_data, test_data,
#               'results/approximation/approximation_avg_mse_train_data_1.txt')
# mse_avg_table(epoch, epsilon, l_rate, m_rate, train_data2, test_data,
#               'results/approximation/approximation_avg_mse_train_data_2.txt')

l_rate = 0.01
m_rate = 0.005
# Plot approximation throughout epochs for training data 1
time_jump = 400
plt.title("Approximation throughout epochs\n" +
          "(training data 1)" + '\nlearning rate = ' + str(l_rate) + ', momentum = ' + str(m_rate))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=.4, linestyle='--')
network4 = nn.NeuralNetwork(n_of_inputs, 6, n_of_outputs, 1, l_rate, m_rate, approximation, 1)
predict(network4)
plt.plot(x, predict(network4), c=color[0], label=str(0) + " epoch")
for j in range(1, 6):
    network4.train_network(train_data, time_jump, epsilon)
    plt.plot(x, predict(network4), c=color[j], label=str(j * time_jump) + " epoch", linewidth=1)
plt.scatter(train_data[:, 0], train_data[:, 1], c="lightgreen", label="Train data")
plt.legend()
plt.savefig('results/approximation/approximation_throughout_epochs_data_1.png')
plt.clf()