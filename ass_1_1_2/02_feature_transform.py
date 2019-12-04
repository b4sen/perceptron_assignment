import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from perc_train import percTrain

np.random.seed(1337)
plt.style.use("seaborn-whitegrid")

# Load the dataset
df = pickle.load(open('../ass_1_1_1/mnist_df_scaled.pickle', 'rb'))  # load the normalized dataset from the previous task
df.columns = ['x1', 'x2', 't']  # rename cols
df.replace({'t': 0}, -1, inplace=True)  # replace 0 with -1

# Load the test dataset
test_df = pickle.load(open('../ass_1_1_1/mnist_test_scaled.pickle', 'rb'))  # load the normalized TEST dataset
test_df.columns = ['x1', 'x2', 't']
test_df.replace({'t': 0}, -1, inplace=True)


def featureTransform(df):
    ft = []
    for i in range(len(df)):
        x1 = df[i][0]
        x2 = df[i][1]
        t = df[i][2]
        arr = [1, x1, x2, x1**2, x2**2, x1 * x2, t]
        ft.append(arr)
    return pd.DataFrame(ft, columns=['1', 'x1', 'x2', 'x1sq', 'x2sq', 'x1x2', 't'])


# Sign function
def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


# pass in the input vectors
def predict(x, y):
    arr = []
    for i in range(len(x)):
        sol = np.dot(y, x[i])
        arr.append(sign(sol))
    return arr


# Transform the input variables of the meshgrid
def transform(x):
    arr = []
    for i in range(len(x)):
        x1 = x[i][0]
        x2 = x[i][1]
        arr.append([1, x1, x2, x1**2, x2**2, x1 * x2])
    return arr


def perc(W, X):
    return sign(np.dot(W, X))


# CHECK ERROR RATE
def err_rate(DF, w):
    sum = 0
    X = DF.iloc[:, :-1].values
    t = DF.iloc[:, -1].values
    for i in range(len(X)):
        pred = perc(w, X[i])
        if pred != t[i]:
            sum += 1

    if sum > 0:
        print("Error rate: {}".format(sum / len(X)))
        print(f"{sum} out of {len(X)} have been falsely classified.")
    else:
        print("All inputs were correctly classified.")



# Transformed DF
ft_df = featureTransform((df.values))
ft_test_df = featureTransform(test_df.values)

# Transformed W Vector
transformed_weight_online = percTrain(ft_df.iloc[:, :6], ft_df.iloc[:, -1], 100)
transformed_weight_batch = percTrain(ft_df.iloc[:, :6], ft_df.iloc[:, -1], 1200, False)

# Creating meshgrid
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
xx, yy = np.meshgrid(x, y)
rav = np.c_[xx.ravel(), yy.ravel()]

transformed_meshgrid = np.array(transform(rav))

# Predicting grid values using the weights from the online algorithm
z_onl = predict(transformed_meshgrid, transformed_weight_online)
z_onl = np.reshape(np.array(z_onl), xx.shape)

# Predicting grid values using the weights from the batch algorithm
z_batch = predict(transformed_meshgrid, transformed_weight_batch)
z_batch = np.reshape(np.array(z_batch), xx.shape)


# Error Rate
print("Online error rate:")
err_rate(ft_df, transformed_weight_online)

print("Batch error rate:")
err_rate(ft_df, transformed_weight_batch)

print('Online error rate on test df:')
err_rate(ft_test_df, transformed_weight_online)

print('Batch error rate on test df:')
err_rate(ft_test_df, transformed_weight_batch)


# Plotting the results
fig = plt.figure(figsize=(12, 9))

fig.add_subplot(2, 2, 1)
plt.contourf(xx, yy, z_batch, alpha=0.3)
plt.scatter(df.x1[df['t'] == 1], df.x2[df['t'] == 1], c='r', alpha=0.3)
plt.scatter(df.x1[df['t'] == -1], df.x2[df['t'] == -1], c='b', alpha=0.3)
plt.title("Batch training")

fig.add_subplot(2, 2, 3)
plt.contourf(xx, yy, z_onl, alpha=0.3)
plt.scatter(df.x1[df['t'] == 1], df.x2[df['t'] == 1], c='r', alpha=0.3)
plt.scatter(df.x1[df['t'] == -1], df.x2[df['t'] == -1], c='b', alpha=0.3)
plt.title("Online training")


fig.add_subplot(2, 2, 2)
plt.contourf(xx, yy, z_batch, alpha=0.3)
plt.scatter(test_df.x1[test_df['t'] == 1], test_df.x2[test_df['t'] == 1], c='r', alpha=0.3)
plt.scatter(test_df.x1[test_df['t'] == -1], test_df.x2[test_df['t'] == -1], c='b', alpha=0.3)
plt.title("Batch training on test")

fig.add_subplot(2, 2, 4)
plt.contourf(xx, yy, z_onl, alpha=0.3)
plt.scatter(test_df.x1[test_df['t'] == 1], test_df.x2[test_df['t'] == 1], c='r', alpha=0.3)
plt.scatter(test_df.x1[test_df['t'] == -1], test_df.x2[test_df['t'] == -1], c='b', alpha=0.3)
plt.title("Online training on test")

plt.show()

"""
Online error rate:
Error rate: 0.005
5 out of 1000 have been falsely classified.
Batch error rate:
Error rate: 0.005
5 out of 1000 have been falsely classified.
Online error rate on test df:
Error rate: 0.001
1 out of 1000 have been falsely classified.
Batch error rate on test df:
Error rate: 0.004
4 out of 1000 have been falsely classified.

"""
