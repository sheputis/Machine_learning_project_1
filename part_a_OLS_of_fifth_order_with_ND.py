from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
# in this we add normal random variable of a variance zigma and calculate the squared error and r2_score
def FrankeFunction(x,y,noise=0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05) #0.05
n = y.size

one_vec = x*0+1
zigma = 0.1 #this is the variance of the noise var(y)
noise = zigma*np.random.randn(n,n)

f=FrankeFunction(x,y,noise)

#beta_lin_reg = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(f)
x, y = np.meshgrid(x,y) #the x's and y's are the matrices that will be plotted
x_, y_,noise_ = x.reshape(-1, 1), y.reshape(-1,1), noise.reshape(-1,1) #the x_'s and y_'s will be the vectors used for calculation

X_ = np.c_[x_,y_]

poly = PolynomialFeatures(5)
X_ = poly.fit_transform(X_) #generating the sample matrix with two variables that are both polynomials of 5th order




z = FrankeFunction(x, y,noise)
z_= FrankeFunction(x_, y_,noise_)
beta_lin_reg = (np.linalg.inv(X_.T.dot(X_)).dot(X_.T)).dot(z_)


z_fit_ = X_.dot(beta_lin_reg)
z_fit=z_fit_.reshape((20,20))

print("_____________________________calculating variance in beta variables________________________________________")
var_Beta = (np.linalg.inv(X_.T.dot(X_)))*zigma
for i in range(21): #writing out variances
    print(var_Beta[i][i])
print("Mean squared error: %.5f" % mean_squared_error(z_,z_fit_))
print("R2r2_score: %.5f" % r2_score(z_,z_fit_))
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(x,y,z_fit, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
