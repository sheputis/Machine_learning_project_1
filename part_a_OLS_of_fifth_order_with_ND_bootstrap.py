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

print("____________________________________________errors_________________________________________________________")
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

#plt.show()

print("____________________________________________BOOTSTRAP________________________________________________________")

class bootstrap:
    def __init__(self,original_array_x,original_array_y):
        self.original_array_x = original_array_x
        self.original_array_y = original_array_y
        self.n = len(original_array_x)
        self.n_test = None;
        self.array_of_indices_training = self.generate_training_indices()
        self.array_of_indices_test = self.generate_test_indices()
        self.training_data_x , self.training_data_y = self.generate_training_data()
        self.test_data_x , self.test_data_y = self.generate_test_data()

    def generate_training_indices(self):
        return np.random.random_integers(self.n,size = self.n)-1 #indices from 0 to n-1

    def generate_test_indices(self):
        test_indices =[]
        for i in range(self.n):
            if sum(self.array_of_indices_training==i)==0:
                test_indices.append(i)
        test_indices = np.array(test_indices)
        return test_indices

    def generate_training_data(self):
        temp_x = self.original_array_x.copy()
        temp_y = self.original_array_y.copy()
        for i in range(self.n):
            temp_x[i] = self.original_array_x[self.array_of_indices_training[i]]
            temp_y[i] = self.original_array_y[self.array_of_indices_training[i]]
        return temp_x , temp_y
    def generate_test_data(self):
        self.n_test = len(self.array_of_indices_test)
        temp_x = self.original_array_x[:self.n_test].copy()
        temp_y = self.original_array_x[:self.n_test].copy()
        for i in range(self.n_test):
            temp_x[i] = self.original_array_x[self.array_of_indices_test[i]]
            temp_y[i] = self.original_array_y[self.array_of_indices_test[i]]
        return temp_x,temp_y

a=np.array([[1],[2],[3],[4],[5]])
b=np.array([[10],[20],[30],[40],[50]])
print("c.original_array_x")
print(a)
c = bootstrap(a,b)
c.generate_training_indices()
print("c.array_of_indices_training")
print(c.array_of_indices_training)
print("c.array_of_indices_test")
print(c.array_of_indices_test)
print("c.training_data_x")
print(c.training_data_x)
print("c.training_data_y")
print(c.training_data_y)
print("c.test_data_x")
print(c.test_data_x)
