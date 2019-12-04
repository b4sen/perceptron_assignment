import pickle
import numpy as np
import matplotlib.pyplot as plt
from perc_train import percTrain

np.random.seed(1337)
plt.style.use("seaborn-whitegrid")

df = pickle.load(open('../ass_1_1_1/mnist_df_scaled.pickle', 'rb'))  # load the normalized dataset from the previous task

df.columns = ['x1', 'x2', 't']  # rename cols
df.replace({'t': 0}, -1, inplace=True)  # replace 0 with -1
df['x0'] = 1  # adding x0 to create homogenous input coordinates
df = df.reindex(columns=['x0', 'x1', 'x2', 't'])  # swap the columns


test_df = pickle.load(open('../ass_1_1_1/mnist_test_scaled.pickle', 'rb'))  # load the normalized TEST dataset
test_df.columns = ['x1', 'x2', 't']
test_df.replace({'t': 0}, -1, inplace=True)
test_df['x0'] = 1
test_df = test_df.reindex(columns=['x0', 'x1', 'x2', 't'])


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def perc(W, X):
    return sign(np.dot(W, X))


# CHECK ERROR RATE
def err_rate(DF, w):
    sum = 0
    X = DF.iloc[:, :3].values
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


# Plot the decision boundary, taken from the following link:
# https://stackoverflow.com/a/48022752
def plot_data(df, inputs, targets, weights):

    # plot input samples(2D data points) and i have two classes.
    plt.scatter(df.x1[df['t'] == 1], df.x2[df['t'] == 1], c='r', alpha=0.3)
    plt.scatter(df.x1[df['t'] == -1], df.x2[df['t'] == -1], c='b', alpha=0.3)

    # Here i am calculating slope and intercept with given three weights
    linsp = np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1]))
    sol = []
    for i in linsp:
        slope = -(weights[0] / weights[2]) / (weights[0] / weights[1])
        intercept = -weights[0] / weights[2]

        #y =mx+c, m is slope and c is intercept
        y = (slope * i) + intercept
        sol.append(y)
    plt.plot(linsp, sol, linewidth=4, c='k')
    plt.xlabel("Edge count")
    plt.ylabel("Area of the smallest enclosing rectangle")


weights = percTrain(df.iloc[:, :3], df.iloc[:, -1], 100)  # online algorithm
batch = percTrain(df.iloc[:, :3], df.iloc[:, -1], 1000, False)  # batch algorithm

print('Error rate of the training dataset:\n')
err_rate(df, batch)
err_rate(df, weights)

# ERROR RATE ON THE TEST DF
print('Error rate of the test dataset:\n')
err_rate(test_df, batch)
err_rate(test_df, weights)

fig = plt.figure(figsize=(12, 9))


fig.add_subplot(2, 2, 1)
plot_data(df, df.iloc[:, 1:3].values, df.iloc[:, -1].values, batch)
plt.title("Batch training algorithm")


fig.add_subplot(2, 2, 3)
plot_data(df, df.iloc[:, 1:3].values, df.iloc[:, -1].values, weights)
plt.title("Online training algorithm")

fig.add_subplot(2, 2, 2)
plot_data(test_df, test_df.iloc[:, 1:3].values, test_df.iloc[:, -1].values, batch)
plt.title('Batch plot on test df')

fig.add_subplot(2, 2, 4)
plot_data(test_df, test_df.iloc[:, 1:3].values, test_df.iloc[:, -1].values, weights)
plt.title('Online plot on test df')


plt.show()


"""
Error rate of the training dataset:

Error rate: 0.004
4 out of 1000 have been falsely classified.
Error rate: 0.004
4 out of 1000 have been falsely classified.


Error rate of the test dataset:

Error rate: 0.003
3 out of 1000 have been falsely classified.
Error rate: 0.001
1 out of 1000 have been falsely classified.

"""
