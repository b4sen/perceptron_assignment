import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

np.random.seed(123)

sc_input = np.arange(0, 5.1, 0.1)
G = 15


def y(x, g):
    arr = []
    for i in x:
        arr.append(2 * (i**2) - g * i + 1)
    return arr


def transform(x, d):
    eq = []
    for i in range(d + 1):
        eq.append(x**i)
    return np.array(eq)


labels = y(sc_input, G)

mu, sigma = 0, 4
N = np.random.normal(loc=0, scale=4, size=7)

x = sc_input[::8]
y = labels[::8] + N


# create a transformed dataset
trans_x_2 = [transform(i, 2) for i in x]
trans_x_4 = [transform(i, 4) for i in x]


def lmsTrain(X, y, epochs=10000, lr=0.001):
    #weights = np.c_[1, np.random.uniform(-1, 1, (1, len(X[0]) - 1))][0]
    weights = np.zeros(len(X[0]))
    for i in range(epochs):
        for j in range(len(X)):
            o = np.dot(weights, X[j])
            err = (y[j] - o)
            #print("Error for X{} :{}".format(i, err))
            weights += err * lr * X[j]
    return weights


def predict(X, w):
    predictions = []
    for i in X:
        predictions.append(np.dot(w, i))
    return predictions


# Train the polynom to the 2nd power
w_2 = lmsTrain(trans_x_2, y)
pred_2 = predict(trans_x_2, w_2)

# Train the polynom to the 4th power
w_4 = lmsTrain(trans_x_4, y, epochs=100000, lr=0.00001)
pred_4 = predict(trans_x_4, w_4)


plt.plot(range(1, 8), pred_2)
plt.plot(range(1, 8), pred_4)
plt.plot(range(1, 8), labels[::8])
plt.scatter(range(1, 8), y)
plt.legend(['Prediction 2', 'Prediction 4', 'Function', 'Target'])
plt.show()
print(w_2)
print(w_4)

"""
[ -1.99539446 -10.7326555    0.91943202]
[-2.2844059  -6.54276158 -8.214805    4.26704386 -0.54222652]
"""
