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

filename = str("data/approx_2.csv")
with open(filename, 'r') as inFile:
    approx_2 = np.loadtxt(filename, delimiter=" ")

filename = str("data/approx_test.csv")
with open(filename, 'r') as inFile:
    approx_test = np.loadtxt(filename, delimiter=" ")

approx_test.sort(0)
approx_1.sort(0)
# Approx
approx_1_train_x = approx_1[:, 0, np.newaxis]
approx_1_train_y = approx_1[:, 1, np.newaxis]


# fitting RBF-Network with data
model = rbf.RBF(hidden_shape=10, sigma=1.)

model.fit_two_stages(approx_1_train_x, approx_1_train_y)
y_pred = model.predict(approx_test[:, 0])

plt.scatter(approx_1_train_x, approx_1_train_y, c='red', s=2, label='real')
plt.plot(approx_test[:, 0], y_pred, 'b-', label='fit')
plt.legend(loc='upper right')
plt.title('Approximation')
plt.show()
plt.clf()
plt.title('Approximation test')
# random test data
temp = np.linspace(-2, 3, 100)
x2 = temp[:, np.newaxis]
y_pred2 = model.predict(x2)
print(y_pred2)

plt.scatter(approx_1_train_x, approx_1_train_x, c='red', s=2, label='real')
plt.plot(x2, y_pred2, 'b-', label='linspace')
plt.legend(loc='upper right')
plt.title('Approximation')
plt.show()