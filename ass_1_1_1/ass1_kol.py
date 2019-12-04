from keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


random.seed(1)

(trainX, trainY), (testX, testY) = mnist.load_data()

train_elems_1 = random.choices([img for i, img in enumerate(trainX) if trainY[i] == 1], k=500)
train_elems_0 = random.choices([img for i, img in enumerate(trainX) if trainY[i] == 0], k=500)

test_elems_1 = random.choices([img for i, img in enumerate(testX) if testY[i] == 1], k=500)
test_elems_0 = random.choices([img for i, img in enumerate(testX) if testY[i] == 0], k=500)


# Canny edge detection -> returns the count of nonzero values
def cnt_edges(img):
    blur = cv2.blur(img, (2, 2))
    blur = cv2.medianBlur(blur, 3)
    edges = cv2.Canny(blur, 100, 255)
    return np.count_nonzero(edges)


# calculates the area of the smallest rectangle drawn around the number
def calc_minrec(img):
    blur = cv2.blur(img, (4, 4))
    #_, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_TRUNC)
    cnts, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0].reshape(-1, 2)
    rect = cv2.minAreaRect(cnts)
    box = cv2.boxPoints(rect)
    x_arr = []
    y_arr = []
    for x, y in box:
        x_arr.append(x)
        y_arr.append(y)
    return polyArea(x_arr, y_arr)


# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates/29590740#29590740
def polyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


#edge_ones = [cnt_edges(img) for img in train_elems_1]
edge_ones = list(map(cnt_edges, train_elems_1)) # trying out different approaches
edge_zeroes = [cnt_edges(img) for img in train_elems_0]

area_ones = [calc_minrec(img) for img in train_elems_1]
area_zeroes = [calc_minrec(img) for img in train_elems_0]


test_edge_1 = [cnt_edges(img) for img in test_elems_1]
test_edge_0 = [cnt_edges(img) for img in test_elems_0]

test_area_1 = [calc_minrec(img) for img in test_elems_1]
test_area_0 = [calc_minrec(img) for img in test_elems_0]

edge_list = {'zeroes': edge_zeroes, 'ones': edge_ones}
edge_df = pd.DataFrame(data=edge_list)

area_list = {'zeroes': area_zeroes, 'ones': area_ones}
area_df = pd.DataFrame(data=area_list)

# CREATE A DF FROM THE TEST CASES
test_edge_list = {'zeros': test_edge_0, 'ones': test_edge_1}
test_edge_df = pd.DataFrame(data=test_edge_list)

test_area_list = {'zeros': test_area_0, 'ones': test_area_1}
test_area_df = pd.DataFrame(data=test_area_list)


# CREATE FIGURE TO PLOT
fig = plt.figure(figsize=(8, 8))
fig.suptitle('TU WIEN MLVC Assignment 1.1.1')

# plot EDGE COUNT
plt.subplot2grid((3, 2), (0, 0))
plt.boxplot(x=[edge_zeroes, edge_ones], labels=list('01'))
plt.title('Edges')

# plot SMALLEST ENCLOSING RECTANGLE
plt.subplot2grid((3, 2), (0, 1))
plt.boxplot(x=[area_zeroes, area_ones], labels=list('01'))
plt.title('Smallest Rectangle Area')

plt.subplot2grid((3, 2), (1, 0))
sns.distplot(edge_df[['zeroes']], hist=False, rug=False, label='Zeroes')
sns.distplot(edge_df[['ones']], hist=False, rug=False, label='Ones')
plt.legend()
plt.title('Edges Dist')

plt.subplot2grid((3, 2), (1, 1))
sns.distplot(area_df[['zeroes']], hist=False, rug=False, label='Zeroes')
sns.distplot(area_df[['ones']], hist=False, rug=False, label='Ones')
plt.legend()
plt.title('Rect Dist')


plt.subplot2grid((3, 2), (2, 0), colspan=2)
plt.scatter(edge_df[['zeroes']], area_df[['zeroes']], c='b', marker='o', label='Zeroes', alpha=0.5)
plt.scatter(edge_df[['ones']], area_df[['ones']], c='r', marker='*', label='Ones', alpha=0.5)
plt.title('Edges to Rect Area', y=-0.01)


"""
plt.subplot2grid((3, 2), (2, 0), colspan=2)
plt.scatter(test_edge_df[['zeros']], test_area_df[['zeros']], c='b', marker='o', label='Zeroes', alpha=0.5)
plt.scatter(test_edge_df[['ones']], test_area_df[['ones']], c='r', marker='*', label='Ones', alpha=0.5)
plt.title('Edges to Rect Area', y=-0.01)
"""

plt.show()

"""
dict_one = {'area': area_ones, 'edges': edge_ones, 'label': [1 for i in range(len(edge_ones))]}
dict_zero = {'area': area_zeroes, 'edges': edge_zeroes, 'label': [0 for i in range(len(edge_zeroes))]}

df = pd.DataFrame(data=dict_one)
df_temp = pd.DataFrame(data=dict_zero)

df = df.append(df_temp, ignore_index=True)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle rows and reset index

pickle.dump(df, open('mnist_df.pickle', 'wb'))

_norm_df = min_max_scaler.fit_transform(df.values)
_df = pd.DataFrame(_norm_df)

pickle.dump(_df, open('mnist_df_scaled.pickle', 'wb'))


# CREATE THE NORMALIZED TEST DF
test_dict_1 = {'area': test_area_1, 'edges': test_edge_1, 'label': [1 for i in range(len(test_edge_1))]}
test_dict_0 = {'area': test_area_0, 'edges': test_edge_0, 'label': [0 for i in range(len(test_edge_0))]}

test_df = pd.DataFrame(data=test_dict_1)
test_temp = pd.DataFrame(data=test_dict_0)
test_df = test_df.append(test_temp, ignore_index=True)
test_df = test_df.sample(frac=1, random_state=1).reset_index(drop=True)
_test_norm = min_max_scaler.fit_transform(test_df.values)
_test_df = pd.DataFrame(_test_norm)
pickle.dump(_test_df, open('mnist_test_scaled.pickle', 'wb'))
"""
