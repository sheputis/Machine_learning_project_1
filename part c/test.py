import numpy as np
import scipy.linalg as scl
import sklearn.linear_model as sklm
from sklearn.preprocessing import PolynomialFeatures

n = 1000
x = np.random.random(n)
y = np.random.random(n)

z = 2 * x  + 5



"""
lasso=linear_model.Lasso(alpha=1, fit_intercept=False)#, max_iter=maxIterations)
polyLasso = PolynomialFeatures(5)
XHatLasso = np.c_[A.x_,A.y_]
XHatLasso = polyLasso.fit_transform(XHatLasso)
lasso.fit(XHatLasso, A.z_)
print(lasso.coef_)
"""


"""
# Note the use of scipy.linalg instead of numpy.linalg as Morten points out.
# I don't think this is the reason for the discrepancy you have in your code.
beta1 = scl.inv(X.T.dot(X) + np.identity(21)).dot(X.T).dot(z)

print (beta1)

# Note here that we use fit_intercept=False. This means that Scikit treats your
# data as "centered". In other words, it doesn't append a column of ones to your
# input data X.
clf = sklm.Ridge(alpha=1, fit_intercept=False)
clf.fit(X, z)
print(X.shape)
print(z.shape)
# Should give the same as beta1 above with clf.intercept_ == 0.
print (clf.intercept_, clf.coef_)
"""
X = np.c_[np.ones(n), x, y]
clf = sklm.Lasso(alpha=1, fit_intercept=False)
polyLasso = PolynomialFeatures(2)
XHatLasso = np.c_[x]
XHatLasso = polyLasso.fit_transform(XHatLasso)
clf.fit(XHatLasso, z)
print(X.shape)
print(z.shape)
# Should give the same as beta1 above with clf.intercept_ == 0.
print (clf.intercept_, clf.coef_)
