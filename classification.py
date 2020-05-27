import numpy as np
import rbf
import matplotlib.pyplot as plt
import os

filename = str("data/iris.csv")
data_training = np.genfromtxt(filename, delimiter=',', dtype=['<f8', '<f8', '<f8', '<f8', 'U15'],
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


# fitting RBF-Network with data
model = rbf.RBF(hidden_shape=10, sigma=1.)
data_train, data_test = data_division(data_training, 0.8)
data_train_conv = transform_data(data_train, len(data_train), 4, 3)
data_test_conv = transform_data(data_test, len(data_test), 4, 3)
model.fit(data_train_conv[:, 0:4], data_train_conv[:, 4:7])
y_predict = model.predict(data_test_conv[:, 0:4])
result = 0
for i in range(len(data_test_conv)):
    origin = change_to_iris_name(y_predict[i], 0.80)
    pred = change_to_iris_name(data_test_conv[i, 4:7], 0.80)
    if origin == pred:
        result += 1
    print(data_test_conv[i, 0:4], " : ", origin, " = ", pred)
result = result / len(data_test_conv) * 100
print('Result: ', result, "%")
