from ML_classes import *
"""
lambda_list=[0,0.1,2,6,20]
for lamd in lambda_list:
    bias=[]
    variance=[]
    for deg in range(9):
        print("running bootstrap of degree %s and lamda %s" % ((deg+1),lamd))
        A = Ridge_main(deg+1,lamd)
        B = run_the_bootstraps(A.x_,A.y_,A.z_,deg+1,lamd)
        C = var_and_bias(A.X_,A.z_,B.beta_list)
        print('bias')
        print(C.bias)
        print('variance')
        print(C.var)
        bias.append(C.bias)
        variance.append(C.var)
    bias=np.array(bias)
    variance=np.array(variance)
    #plt.plot(bias,label ='bias %d' % lamd)
    #plt.plot(variance, label ='variance')
    plt.plot(variance+bias, label ='lamda %d' % lamd)



plt.legend()
plt.show()
"""
lamd = 0
A = Ridge_main(5,lamd)
X_train = A.X_
y_train = A.z_



import numpy as np
import scipy.linalg as scl
import sklearn.linear_model as sklm
"""
n = 1000
x = np.random.random(n)
y = np.random.random(n)

z = 2 * x + y + 5

X = np.c_[np.ones(n), x, y]"""

X=A.X_
z=A.z_
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
clf = sklm.Lasso(alpha=1, fit_intercept=False)
clf.fit(X, z)
print(X.shape)
print(z.shape)
# Should give the same as beta1 above with clf.intercept_ == 0.
print (clf.intercept_, clf.coef_)
