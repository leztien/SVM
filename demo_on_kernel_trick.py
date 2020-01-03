
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


from itertools import product
g = product(X,X)
t = tuple(g)
b = 0
dot_products1 = [(np.dot(t[0],t[1])+b)**degree for t in t]
dot_products1 = minmax(dot_products1)

g = product(Xpoly, Xpoly)
t = tuple(g)
dot_products2 = [np.dot(t[0],t[1]) for t in t]
dot_products2 = minmax(dot_products2)
r = dot_products1.mean() - dot_products2.mean()
dot_products2 = dot_products2 + r

plt.plot(dot_products1, marker='.', alpha=0.5)
plt.plot(dot_products2, marker='.', alpha=0.5)

t = np.sort(tuple(np.quantile(dot_products1, q=[0.25,0.75])) + tuple(np.quantile(dot_products2, q=[0.25,0.75])))[[0,-1]]
plt.ylim(t[0]-0.015, t[1]+0.015)


