import numpy as np
import pandas as pd
from numpy.linalg import lstsq, inv
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

np.random.seed(123)

x = np.arange(start=0, stop=5.1, step=0.1)

G = 15
y = 2 * x**2 - G * x + 1

distr = np.random.normal(loc=0, scale=4, size=7)

test_x = x[0::8]
t = y[0::8] + distr


def fi_func(x, d=5):
    vec = []

    for element in x:
        vec.append([element**i for i in range(0, d + 1)])
        #print("the element is: ", element)
        #print("what was appended: ", [element**i for i in range(0, d + 1)])

    return(vec)


"""
print("test_x:")
print(test_x, "\n")
print()

print(np.empty_like([len(test_x), 3]))
"""

fi_2 = fi_func(test_x, 2)
fi_4 = fi_func(test_x, 4)


min_w_2 = np.dot(np.dot(inv(np.dot(np.transpose(fi_2), fi_2)), np.transpose(fi_2)), t)  # definition of least sqaures
# print(min_w_2)

min_w_4 = np.dot(np.dot(inv(np.dot(np.transpose(fi_4), fi_4)), np.transpose(fi_4)), t)

res = lstsq(fi_2, t)[0]  # another way to find the least sqaures
# print(res)


pred_2 = [np.dot(min_w_2, x) for x in fi_2]
pred_4 = [np.dot(min_w_4, x) for x in fi_4]


plt.plot(range(1, 8), pred_2)
plt.plot(range(1, 8), pred_4)
plt.plot(range(1, 8), y[0::8])
plt.scatter(range(1, 8), t)
plt.legend(['Predicted 2', 'Predicted 4', 'Function', 'Target'])
plt.title("LMS Algorithm\nWith G = 15")

plt.show()

print(min_w_2)
print(min_w_4)

"""
[ -1.52590358 -11.83527362   1.20762611]
[ -3.76332302  21.80646235 -37.42319977  13.39455318  -1.4208908 ]
"""
