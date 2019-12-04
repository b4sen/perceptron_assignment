from keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

random.seed(1)

(trainX, trainY), (testX, testY) = mnist.load_data()

train_elems_1 = random.choices([img for i, img in enumerate(trainX) if trainY[i] == 1], k=500)
train_elems_0 = random.choices([img for i, img in enumerate(trainX) if trainY[i] == 0], k=500)

test_elems_1 = random.choices([img for i, img in enumerate(testX) if testY[i] == 1], k=500)
test_elems_0 = random.choices([img for i, img in enumerate(testX) if testY[i] == 0], k=500)

# prepare the dataframes and concatenate
df = pd.DataFrame(pd.Series(train_elems_1))
temp_df = pd.DataFrame(pd.Series(train_elems_0))

df['t'] = 1
temp_df['t'] = -1

df = df.append(temp_df, ignore_index=True)
df.columns = ['x', 't']
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

#### PREPARE THE TEST DATAFRAME
test_df = pd.DataFrame(pd.Series(test_elems_1))
test_temp = pd.DataFrame(pd.Series(test_elems_0))

test_df['t'] = 1
test_temp['t'] = -1

test_df = test_df.append(test_temp, ignore_index=True)
test_df.columns = ['x', 't']
test_df = test_df.sample(frac=1, random_state=1).reset_index(drop=True)


def percTrain(X, t, maxIts, online=True):
    cnt = 0
    bool_arr = [False for i in range(len(X))]
    if online:
        w = np.zeros(X.values[0].ravel().shape)  # change the first value if there is a bias given
        lr = 1
        # online algorithm:
        # adapt the weight vector constantly
        while not all(bool_arr) and cnt < maxIts:
            for i in range(len(X)):
                xi = X.values[i].ravel()  # since we have the whole image in 1 column, expand it
                ti = t.values[i]
                wi = np.array(xi) * ti
                xt = np.dot(w, wi)
                if xt <= 0.0:
                    w += wi * lr
                    bool_arr[i] = False
                else:
                    bool_arr[i] = True
            cnt += 1
    else:
        #w = np.zeros(X.values[0].ravel().shape)
        w = np.c_[1, np.random.uniform(-1, 1, (1, len(X.values[0].ravel()) - 1))][0]  # randomly initialized weight vector
        lr = 0.001
        # do batch optimization: update weight after each epoch
        while not all(bool_arr) and cnt < maxIts:
            delta_w = np.zeros(X.values[0].ravel().shape)
            for i in range(len(X)):
                xi = X.values[i].ravel()
                ti = t.values[i]
                wi = np.array(xi) * ti
                xt = np.dot(w, wi)
                if xt <= 0.0:
                    delta_w += wi
                    bool_arr[i] = False
                else:
                    bool_arr[i] = True
            w += delta_w * lr
            cnt += 1
    return w


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def perc(W, X):
    return sign(np.dot(W, X))


# CHECK ERROR RATE
def err_rate(DF, w):
    sum = 0
    X = DF.iloc[:, 0].values
    t = DF.iloc[:, -1].values
    for i in range(len(X)):
        pred = perc(w, X[i].ravel())
        if pred != t[i]:
            sum += 1

    if sum > 0:
        print("Error rate: {}".format(sum / len(X)))
        print(f"{sum} out of {len(X)} have been falsely classified.")
    else:
        print("All inputs were correctly classified.")


np.random.seed(1)
# Calculate weights
weights_online = percTrain(df.iloc[:, 0], df.iloc[:, -1], 1000)
weights_batch = percTrain(df.iloc[:, 0], df.iloc[:, -1], 1000, False)

print("Online error rate:")
err_rate(df, weights_online)
print(" ")
print("Batch error rate:")
err_rate(df, weights_batch)


print('Online error rate for the test dataset:')
err_rate(test_df, weights_online)
print(' ')
print('Batch error rate for the test dataset:')
err_rate(test_df, weights_batch)


# Reshape the weights to the shape of the original 28*28 MNIST image
w = np.reshape(weights_online, (28, 28))
w_b = np.reshape(weights_batch, (28, 28))


fig = plt.figure(figsize=(6, 9))
fig.add_subplot(2, 1, 2)
plt.imshow(w, 'gray')
plt.title('Weights of the online training')

fig.add_subplot(2, 1, 1)
plt.imshow(w_b, 'gray')
plt.title('Weights of the batch training')

plt.show()


"""
Online error rate:
All inputs were correctly classified.
 
Batch error rate:
All inputs were correctly classified.

Online error rate for the test dataset:
Error rate: 0.003
3 out of 1000 have been falsely classified.
 
Batch error rate for the test dataset:
All inputs were correctly classified.

"""
