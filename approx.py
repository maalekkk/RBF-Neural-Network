import numpy as np
import rbf
import matplotlib.pyplot as plt
import os

# Create target Directory
path = "results/approximation"
try:
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")

# Import text files
filename = str("data/approx_1.csv")
with open(filename, 'r') as inFile:
    approx_1 = np.loadtxt(filename, delimiter=" ")

filename = str("data/approx_test.csv")
with open(filename, 'r') as inFile:
    approx_test = np.loadtxt(filename, delimiter=" ")

approx_test.sort(0)
approx_1.sort(0)
# Approx
approx_1_train_x = approx_1[:, 0, np.newaxis]
approx_1_train_y = approx_1[:, 1, np.newaxis]

# # sample inputs and add noise
# NUM_SAMPLES = 100
# # X = np.random.uniform(0., 1., NUM_SAMPLES)
# X = approx_1[:, 0]
# X = np.sort(X, axis=0)
# noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
# # y = np.sin(2 * np.pi * X) + noise
# y = approx_1[:, 1]
# rbfnet = rbf.RBF(10)
# rbfnet.fit_two_stages(X, y, True)
#
# y_pred = rbfnet.predict(X)
#
# plt.plot(X, y, '-o', label='true')
# plt.plot(X, y_pred, '-o', label='RBF-Net')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
#
# # fitting RBF-Network with data
model = rbf.RBF(hidden_neurons=21, random_centroids=False)

# model.fit_two_stages(approx_1_train_x, approx_1_train_y, True)
# y_pred = model.predict(approx_test[:, 0])

model.fit_two_stages(approx_1_train_x, approx_1_train_y, False)
y_pred = model.predict(approx_test[:, 0])

plt.scatter(approx_1_train_x, approx_1_train_y, c='red', s=2, label='real')
plt.plot(approx_test[:, 0], y_pred, 'b-', label='fit')
plt.legend(loc='upper right')
plt.title('Approximation')
plt.show()
plt.clf()
plt.title('Approximation test')
# # random test data
# temp = np.linspace(-2, 3, 100)
# x2 = temp[:, np.newaxis]
# y_pred2 = model.predict(x2)
# print(y_pred2)
#
# plt.scatter(approx_1_train_x, approx_1_train_x, c='red', s=2, label='real')
# plt.plot(x2, y_pred2, 'b-', label='linspace')
# plt.legend(loc='upper right')
# plt.title('Approximation')
# plt.show()