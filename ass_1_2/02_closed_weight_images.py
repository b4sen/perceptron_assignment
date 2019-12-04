import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

###### DEFINE THE INITIAL DATASET ######
DIM = 29

x = np.arange(start=0, stop=5.1, step=0.1)

G = 15
y = 2 * x**2 - G * x + 1

distr = np.random.normal(loc=0, scale=4, size=7)

test_x = x[0::8]
t = y[0::8] + distr  # y values with noise


def createTestSet(X):
    img_arr = []
    for x in X:
        temp = np.zeros((DIM, DIM))
        m1, m2 = np.random.normal(loc=15, scale=2, size=2)
        for i in range(DIM):
            for j in range(DIM):
                if (i - m1)**2 + (j - m2)**2 < 9 * x**2:
                    temp[i][j] = 1
        img_arr.append(temp)
    return np.array(img_arr)


def flattenArray(X):
    flat_arr = []
    for x in X:
        flat_arr.append(np.array(np.insert(x.ravel(), 0, 0, 0)))

    return np.array(flat_arr)


def plotTestImg(x):
    fig = plt.figure(figsize=(9, 9))
    for i in range(len(x)):
        ax = fig.add_subplot(3, 3, i + 1)
        plt.imshow(x[i], 'gray')
        ax.title.set_text('Image index {}'.format(i))
    plt.suptitle("Images of the dataset")
    plt.show()


train_x_img = createTestSet(test_x)


plotTestImg(train_x_img)


flattened = flattenArray(train_x_img)


w_img = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(flattened), flattened)), np.transpose(flattened)), y[::8])
pred_img = [np.dot(w_img, x) for x in flattened]


def computeErr(x, y):
    sse = 0
    for i in range(len(x)):
        sse += (y[i] - x[i])**2
    return sse / 2


train_full = createTestSet(x)
flattened_full = flattenArray(train_full)

w_full = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(flattened_full), flattened_full)), np.transpose(flattened_full)), y)
pred_full = [np.dot(w_full, x) for x in flattened_full]


print("SSE for the 7 images: ", computeErr(pred_img, y[::8]))
print("SSE for the 51 images: ", computeErr(pred_full, y))

fig = plt.figure(figsize=(7, 9))

fig.add_subplot(2, 1, 1)
plt.plot(range(1, 8), pred_img)
plt.plot(range(1, 8), y[::8])
plt.legend(['Predicted', 'Original'])
plt.title("Values of the 7 images\nSSE: {}".format(computeErr(pred_img, y[::8])))


fig.add_subplot(2, 1, 2)
plt.plot(range(1, len(pred_full) + 1), pred_full)
plt.plot(range(1, len(pred_full) + 1), y)
plt.legend(['Predicted', 'Original'])
plt.title("Values of the 51 images\nSSE: {}".format(computeErr(pred_full, y)))


plt.show()
