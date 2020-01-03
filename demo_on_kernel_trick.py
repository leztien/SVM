
"""
visual demo on (polynomial) kernel trick
"""

import numpy as np, matplotlib.pyplot as plt

def minmax(a):
    r = max(a)-min(a)
    a = (a-min(a))/r
    return(a)

def poly(X, degree=3):
    Xpoly = np.zeros(shape=(X.shape[0], degree*X.shape[1]))
    for i in range(1, degree+1):
        Xpoly[:, ((i-1)*X.shape[1]):((i-1)*X.shape[1]+X.shape[1])] = X**i
    return(Xpoly)
    
#############################################################################

X = np.random.randn(10,2)

degree = 5
Xpoly = poly(X, degree=degree)

#
from itertools import product
g = product(X,X)
t = tuple(g)
dot_products1 = [(np.dot(t[0],t[1])+1)**degree for t in t]
dot_products1 = minmax(dot_products1)
nx = np.argsort(dot_products1)
nx = nx[5:-5]
dot_products1 = dot_products1[nx]

#
g = product(Xpoly, Xpoly)
t = tuple(g)
dot_products2 = [np.dot(t[0],t[1]) for t in t]
dot_products2 = minmax(dot_products2)
dot_products2 = dot_products2[nx]

#
from sklearn.preprocessing import PolynomialFeatures
Xpoly = PolynomialFeatures(degree=degree, include_bias=True, interaction_only=False).fit_transform(X)
g = product(Xpoly, Xpoly)
t = tuple(g)
dot_products3 = [np.dot(t[0],t[1]) for t in t]
dot_products3 = minmax(dot_products3)
dot_products3 = dot_products3[nx]


plt.plot(dot_products1, marker='.', alpha=0.5, color='r')
plt.plot(dot_products2, marker='.', alpha=0.5, color='g')
plt.plot(dot_products3, marker='.', alpha=0.5, color='b')

t = np.sort(tuple(np.quantile(dot_products1, q=[0.25,0.75])) + 
            tuple(np.quantile(dot_products2, q=[0.25,0.75])) + 
            tuple(np.quantile(dot_products3, q=[0.25,0.75])))[[0,-1]]
plt.ylim(t[0]-0.015, t[1]+0.015)


