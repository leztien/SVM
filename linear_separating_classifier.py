
"""
this is a simple algorithm which separates two classes linearly
Luis Serrano
https://www.youtube.com/watch?v=Lpr__X8zuE8
"""

import numpy as np
from random import randint

#random state
seed = None
seed = seed or randint(0,1000)
print("seed =", seed)
np.random.seed(seed)

#data
m,n = 100, 2   # chnage n (n_features) here
X = np.random.random(size=(m,n))

#labels (ground truth)
if n==2:
    random = randint(0,1)
    f = lambda x,y : ((x+y-1)if random==1 else (x-y-0)) >= 0 
    y = [int(f(x,y)) for x,y in X]
else:
    nx = X.sum(1).argsort()
    X = X[nx]
    y = np.zeros(shape=m).astype(int)
    y[m//2:] = 1

#add noise (optional)
noise = False
if noise: X += np.random.randn(m,n)*0.1

#initialize weights
w = np.array([0]*n, dtype=float)
b = 0

#the prdict function
predict = lambda x,w,b : np.dot(x,w) + b >= 0

#hyperparameters
η = 0.01
epochs = m*100

#loop
for epoch in range(epochs):
    ix = randint(0, m-1)
    x = X[ix]
    
    #if correctly classified
    ytrue = y[ix]
    ypred = predict(x,w,b)
    if ytrue==ypred:
        continue
    
    #if misclassified
    if (ytrue,ypred) == (0,1):
        w -= η*X[ix]
        b -= η
    else:
        w += η*X[ix]
        b += η
 
#evaluate  
ypred = [predict(x,w,b) for x in X]
acc = np.equal(y,ypred).mean()
print("accuracy =",acc)    

#visualize
if n==2:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["red","navy"])
    plt.scatter(*X.T, c=y, cmap=cmap, s=80, edgecolor='k')
    
    slope = -w[0]/w[1]
    intercept = -b/w[1]
    xline = np.linspace(0,1)
    yline = xline*slope + intercept
    plt.plot(xline, yline)
    plt.xlim(-0.1,1.1); plt.ylim(-.1,1.1)

