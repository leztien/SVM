
"""
Linear Support Vector Classifier with Stochastic Gradient Descent
"""

import numpy as np


def make_data(m=10, n=2, minmax=False, noise=False, seed=None):
    from random import randint
    
    #random state   #692, 998, 377  
    seed = seed or randint(0,1000)
    print("seed =", seed)
    np.random.seed(seed)
    
    #data
    X = np.random.uniform(-1,1, size=(m,n)) 
    
    #labels (ground truth)
    random = randint(0,1)
    if n==2:
        f = lambda x,y : ((x+y-0)if random==1 else (x-y+0)) >= 0 
        y = [int(f(x,y)) for x,y in X]
    else:
        nx = X.sum(1).argsort()
        X = X[nx]
        y = np.zeros(shape=m).astype(int)
        y[m//2:] = 1
    y = [(-1,1)[int(y)] for y in y]
    
    #add a gap between the classes
    mask = np.array(y)==1
    gap = 0.25
    X[mask] += gap if random==1 else (gap, -gap) if n==2 else 0

    #add noise, standerdize, normalize
    if noise: X += np.random.randn(m,n)*0.1
    X = (X-X.mean(0)) / X.std(axis=0, ddof=0)  # data must be centered
    if minmax: X = (X - X.min(0)) / (X.max(0)-X.min(0))
    return(X,y)

###############################################################################

X,y = make_data(m=100, n=2, minmax=False, noise=False, seed=None)


#the algorithm
from random import randint, gauss
C = 1.0
η = 0.01
max_iter = 10000

m,n = X.shape
w = np.array([gauss(mu=0, sigma=1) for _ in range(n)]) * 10
b = 0

for epoch in range(max_iter):
    i = randint(0, m-1)
    xi = X[i]
    yi = y[i]
    if yi*(np.dot(w,xi)+b) >= 1:  #classified correctly
        w -= η*w  # expanding
    else:  # missclassified
        w -= η * (w - C * yi*xi)
        b -= η * (-yi*C)


#learned parameters
print("w,b =", w.round(2), round(b,2))


#evaluate  
predict = lambda x,w,b : (-1,1)[int(np.dot(x,w) + b >= 0)]  #the prdict function
ypred = [predict(x,w,b) for x in X]
acc = np.equal(y,ypred).mean()
print("accuracy =",acc)


#visualize
n = X.shape[1]
if n==2:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["red","navy"])
    plt.scatter(*X.T, c=y, cmap=cmap, s=80, edgecolor='k')
    
    xlim = plt.xlim(); ylim = plt.ylim()
    slope = -w[0]/w[1]
    intercept = -b/w[1]
    xline = np.linspace(*(np.array(xlim)+(-10,10)))
    yline = xline*slope + intercept
    plt.plot(xline, yline)
    plt.plot(xline, yline+1, 'k:')
    plt.plot(xline, yline-1, 'k:')
    plt.xlim(*xlim); plt.ylim(*ylim)


