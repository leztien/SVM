
"""
demo on defining a decision boundry around a 'pathelogical' blob
for the highest recall possible (with SVC with RBF)
"""

import numpy as np, matplotlib.pyplot as plt

def make_data(m=1000, p=0.1, seed=None):
    """2-D data"""
    if seed is True:
        from random import randint
        seed = randint(0, 1000)
        print("random seed =", seed)
    if seed:
        np.random.seed(seed)
    
    (m0,m1) = int(m*(1-p)),int(m*p)
    X0 = np.random.multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]], size=m0)
    X1 = np.random.multivariate_normal(mean=[1,1], cov=[[0.1,0],[0,0.1]], size=m1)
    X = np.vstack([X0,X1])
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    y = np.array([0]*m0 + [1]*m1, dtype="int8")
    return(X,y)

################################################################################
    
#data
X,y = make_data(500, p=0.05, seed=391)  #181, 391
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25, random_state=0)


#model and meta-estimator
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
md = SVC(kernel="rbf")
d = {"C":[0.1, 1, 10, 100, 1000, 1E4], 
     "gamma":[0.01, 0.1, 1, 10]}
metric = "recall"
me = GridSearchCV(md, param_grid=d, scoring=metric, cv=3)
me.fit(Xtrain, ytrain)


#get the best from the meta-estimator
best_parameters = me.best_params_
md = me.best_estimator_
print("best parameters:", best_parameters)


#accuracy
train_acc = md.score(Xtrain, ytrain)
test_acc = md.score(Xtest, ytest)
print("train/test accuracy: {:.0%} / {:.0%}".format(train_acc, test_acc))


#recall bzw accuracy (i.e the chosen cv metric)
d = me.cv_results_
a = d["mean_test_score"]
print("best validation {} = {:.2f}".format(metric, max(a)))


#recall
from sklearn.metrics import recall_score, precision_score
ypred = md.predict(Xtrain)
train_recall = recall_score(ytrain, ypred)
ypred = md.predict(Xtest)
test_recall = recall_score(ytest, ypred)
print("train/test recall: {:.2f} / {:.2f}".format(train_recall, test_recall))


#increasing the recall == lowering the threshold
min_allowed_recall = 0.999
thresholds = np.linspace(0, np.quantile(md.decision_function(Xtrain), q=.75), 100)
recalls = []
zz = md.decision_function(Xtrain)  # the confidence aka z-values of the decision function
for threshold in thresholds:
    ypred = (zz > threshold).astype("uint8")
    recall = recall_score(ytrain, ypred)
    recalls.append(recall)
ix = tuple(np.array(recalls) > min_allowed_recall).index(True)
threshold = thresholds[ix]


#test-set recall and precision
ypred = (md.decision_function(Xtest) > threshold).astype("uint8")
recall = recall_score(ytest, ypred)
precision = precision_score(ytest, ypred)
acc = np.equal(ypred, ytest).mean()
print("test recall / precision / accuracy (after lowering the threshold): {:.2f} / {:.2f} / {:.0%}".format(recall, precision, acc))


#visualize the train set
mask = ytrain==1
plt.plot(*Xtrain[~mask].T, 'o', color='blue', markeredgecolor='k', alpha=0.7)
plt.plot(*Xtrain[mask].T, '^', markersize=7.5, color='red', markeredgecolor='k', alpha=0.6)
plt.axis("equal")

#visualize the margin
r = np.linspace(*(np.sort(X[y==1].ravel())[[0,-1]]+[-1,1]), 100)
XX,YY = np.meshgrid(r,r)
ZZ = md.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
plt.contour(XX,YY,ZZ, levels=[threshold, 0], linestyles=['--','-'], linewidths=[2,4], colors='k', zorder=3)
plt.show()
