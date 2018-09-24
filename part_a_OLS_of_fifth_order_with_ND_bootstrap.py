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


print("____________________________________________BOOTSTRAP________________________________________________________")

class bootstrap:
    def __init__(self,n): #n the length of the array that is put in
        self.n = n
        self.n_test = None;
        self.array_of_indices_training = self.generate_training_indices()
        self.array_of_indices_test = self.generate_test_indices()

    def generate_training_indices(self):
        return np.random.random_integers(self.n,size = self.n)-1 #indices from 0 to n-1

    def generate_test_indices(self):
        test_indices =[]
        for i in range(self.n):
            if sum(self.array_of_indices_training==i)==0:
                test_indices.append(i)
        test_indices = np.array(test_indices)
        return test_indices

    def generate_training_data(self,input_array):
        temp = input_array.copy()
        for i in range(self.n):
            temp[i] = input_array[self.array_of_indices_training[i]]
        return temp
    def generate_test_data(self,input_array):
        self.n_test = len(self.array_of_indices_test)
        temp = input_array[:self.n_test].copy()
        for i in range(self.n_test):
            temp[i] = input_array[self.array_of_indices_test[i]]
        return temp
"""
a=np.array([[1],[2],[3],[4],[5]])
print("c.original_array_x")
print(a)
print('_________')
c = bootstrap(a)
print(c.training_data)
print('_________')
print(c.test_data)
print('_________')
"""
class OLS_main:
    def __init__(self):
        delta=0.05
        x = np.arange(0, 1, delta)
        y = np.arange(0, 1, delta) #0.05
        n = len(x)
        self.x, self.y = np.meshgrid(x,y) #the x's and y's are the matrices that will be plotted
        self.n = self.y.size
        self.zigma = 0.1 #this is the variance of the noise var(y)
        self.noise = self.zigma*np.random.randn(n,n)
        self.x_, self.y_,self.noise_ = self.x.reshape(-1, 1), self.y.reshape(-1,1), self.noise.reshape(-1,1) #the x_'s and y_'s will be the vectors used for calculation
        self.X_ = self.generate_X_of_degree(5)
        self.z  = FrankeFunction(self.x, self.y,self.noise)
        self.z_ = FrankeFunction(self.x_, self.y_,self.noise_)
        self.beta_lin_reg = (np.linalg.inv(self.X_.T.dot(self.X_)).dot(self.X_.T)).dot(self.z_)
        self.z_fit_ = self.X_.dot(self.beta_lin_reg)
        self.z_fit=self.z_fit_.reshape((n,n))


    def generate_X_of_degree(self,n):
        X_ = np.c_[self.x_,self.y_]
        poly = PolynomialFeatures(n)
        return poly.fit_transform(X_) #generating the sample matrix with two variables that are both polynomials of 5th order

    def plot_everything(self):
        print("________________________________________plotting_________________________________________________________")
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.x, self.y, self.z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        surf2 = ax.plot_surface(self.x,self.y,self.z_fit, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def variance_in_beta(self):#this needs to be edited, the number 21 has to be changed to the amount of columns in X (polyfit)
        print("_____________________________calculating variance in beta variables________________________________________")
        var_Beta = (np.linalg.inv(self.X_.T.dot(self.X_)))*self.zigma
        for i in range(21): #writing out variances
            print(var_Beta[i][i])

    def errors(self):
        print("____________________________________________errors_________________________________________________________")
        mse_ = mean_squared_error(self.z_,self.z_fit_)
        r2_score_ = r2_score(self.z_,self.z_fit_)
        print("Mean squared error: %.5f" % mse_)
        print("R2r2_score: %.5f" % r2_score_)
        return mse_ , r2_score_


class OLS_:
    def __init__(self,x,y,z): #x,y are the coordinates and z is the corresponding precalculated(with noise) function output
        self.x = x
        self.y = y
        self.z = z
        self.X = self.generate_X_of_degree(5)
        self.beta = self.find_beta()
        self.z_fit = self.X.dot(self.beta)

    def generate_X_of_degree(self,n):
        X = np.c_[self.x,self.y]
        poly = PolynomialFeatures(n)
        return poly.fit_transform(X) #generating the sample matrix with two variables that are both polynomials of 5th order

    def find_beta(self):
        return (np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T)).dot(self.z)

    def errors(self):
        print("____________________________________________errors_________________________________________________________")
        mse_ = mean_squared_error(self.z,self.z_fit)
        r2_score_ = r2_score(self.z,self.z_fit)
        print("Mean squared error bootstrap: %.5f" % mse_)
        print("R2r2_score bootstrap: %.5f" % r2_score_)
        return mse_ , r2_score_
print("))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))")

def errors(z,z_fit):
    print("____________________________________________errors_________________________________________________________")
    mse_ = mean_squared_error(z,z_fit)
    r2_score_ = r2_score(z,z_fit)
    print("Mean squared error: %.5f" % mse_)
    print("R2r2_score: %.5f" % r2_score_)
    return mse_ , r2_score_

A = OLS_main()
A.errors()
BOOT = bootstrap(len(A.x_))
x_train = BOOT.generate_training_data(A.x_)
y_train = BOOT.generate_training_data(A.y_)
z_train = BOOT.generate_training_data(A.z_)

x_test = BOOT.generate_test_data(A.x_)
y_test = BOOT.generate_test_data(A.y_)
z_test = BOOT.generate_test_data(A.z_)


B = OLS_(x_train,y_train,z_train)
B.errors()
