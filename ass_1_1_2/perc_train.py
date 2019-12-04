import numpy as np

# This is the implementation of the training algorithms:
# It takes an array of inputvectors, the corresponding labels,
# an integer determining the max number of iterations
# and an optional boolean (for the online or batch algorithm) arguments
#
# the output is a weight vector of length 3
def percTrain(X, t, maxIts, online=True):
    cnt = 0
    bool_arr = [False for i in range(len(X))]
    if online:
        # online algorithm:
        # adapt the weight vector constantly
        w = np.zeros(X.values[0].shape)  # change the first value if there is a bias given
        lr = 1
        while not all(bool_arr) and cnt < maxIts:
            for i in range(len(X)):
                xi = X.values[i]
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
        # do batch optimization
        # w = [1, 0.3, -2.1]  # arbitrarily chosen numbers for testing purposes
        w = np.c_[1, np.random.uniform(-1, 1, (1, len(X.values[0]) - 1))][0]  # randomly initialized weight vector
        lr = 0.001
        while not all(bool_arr) and cnt < maxIts:
            delta_w = np.zeros(X.values[0].shape)
            for i in range(len(X)):
                xi = X.values[i]
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
