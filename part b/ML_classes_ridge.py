from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import scipy
# in this we add normal random variable of a variance zigma and calculate the squared error and r2_score
def FrankeFunction(x,y,noise=0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise






print("____________________________________________Ridge_MAIN________________________________________________________")
class Ridge_main:
    def __init__(self,deg,lamd): #deg is degree of the polynomial to be generated
        delta=0.05
        self.deg = deg
        self.lamd = lamd
        x = np.arange(0, 1, delta)
        y = np.arange(0, 1, delta) #0.05
        n = len(x)
        self.x, self.y = np.meshgrid(x,y) #the x's and y's are the matrices that will be plotted
        self.n = self.y.size
        self.zigma = 0.01 #this is the variance of the noise var(y)
        self.noise = self.zigma*np.random.randn(n,n)
        self.x_, self.y_,self.noise_ = self.x.reshape(-1, 1), self.y.reshape(-1,1), self.noise.reshape(-1,1) #the x_'s and y_'s will be the vectors used for calculation
        self.X_ = self.generate_X_of_degree(deg)
        self.z  = FrankeFunction(self.x, self.y,self.noise)
        self.z_ = FrankeFunction(self.x_, self.y_,self.noise_)
        Id = np.identity(self.X_.shape[1])
        self.beta_lin_reg = self.find_beta()
#self.beta_lin_reg = (np.linalg.inv(self.X_.T.dot(self.X_) + self.lamd*Id).dot(self.X_.T)).dot(self.z_)
        self.z_fit_ = self.X_.dot(self.beta_lin_reg)
        #self.z_fit=self.z_fit_.reshape((n,n))

    def find_beta(self):
        Id = np.identity(self.X_.shape[1])
        to_be_inverted = self.X_.T.dot(self.X_) + self.lamd*Id
        U, D, V_T = scipy.linalg.svd(to_be_inverted, full_matrices=False)
        D_inv = 1/D
        D_inv = np.diag(D_inv)
        inverted = V_T.T.dot(D_inv.dot(U.T))
        return (inverted.dot(self.X_.T)).dot(self.z_)

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
        ax.set_xlabel('x axis',fontsize=25)
        ax.set_ylabel('y axis',fontsize=25)
        ax.set_zlabel('z axis',fontsize=25)
        ax.text2D(0.10, 0.95, "part a) Ridge regression, the fitting, lambda = %.2f" % self.lamd , transform=ax.transAxes,fontsize=25)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()



    def variance_in_beta(self):#this needs to be edited, the number 21 has to be changed to the amount of columns in X (polyfit)
        print("_____________________________calculating variance in beta variables_degree_%d________________________________________" % self.deg)
        number = len(self.X_[0]) #of elements in that degree polynomial
        X_T_X= self.X_.T.dot(self.X_)
        lambd_id= np.identity(number)*self.lamd
        el = np.linalg.inv(X_T_X + lambd_id)
        The_matrix =el.dot(X_T_X.dot(el))
        var_Beta = The_matrix*(self.zigma**2)
        for i in range(number): #writing out variances
            print(var_Beta[i][i])

    def errors(self):
    #    print("____________________________________________errors_________________________________________________________")
        mse_ = mean_squared_error(self.z_,self.z_fit_)
        r2_score_ = r2_score(self.z_,self.z_fit_)
    #    print("Mean squared error: %.5f" % mse_)
    #    print("R2r2_score: %.5f" % r2_score_)
        return mse_ , r2_score_
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


class Ridge_:
    def __init__(self,x,y,z,deg,lamd): #x,y are the coordinates and z is the corresponding precalculated(with noise) function output
        self.lamd = lamd
        self.x = x
        self.y = y
        self.z = z
        self.X = self.generate_X_of_degree(deg)
        self.beta = self.find_beta()
        self.z_fit = self.X.dot(self.beta)

    def generate_X_of_degree(self,n):
        X = np.c_[self.x,self.y]
        poly = PolynomialFeatures(n)
        return poly.fit_transform(X) #generating the sample matrix with two variables that are both polynomials of 5th order

    def find_beta(self):
        Id = np.identity(self.X.shape[1])
        return (np.linalg.inv(self.X.T.dot(self.X) + self.lamd*Id).dot(self.X.T)).dot(self.z)

    def errors(self):
        #print("____________________________________________errors_________________________________________________________")
        mse_ = mean_squared_error(self.z,self.z_fit)
        r2_score_ = r2_score(self.z,self.z_fit)
        #print("Mean squared error bootstrap: %.5f" % mse_)
        #print("R2r2_score bootstrap: %.5f" % r2_score_)
        return mse_*len(self.z) , r2_score_
print("))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))")

def errors(z,z_fit):
    print("____________________________________________errors_________________________________________________________")
    mse_ = mean_squared_error(z,z_fit)
    r2_score_ = r2_score(z,z_fit)
    print("Mean squared error: %.5f" % mse_)
    print("R2r2_score: %.5f" % r2_score_)
    return mse_ , r2_score_






print("___________________________calculating many bootstrap mse's ____________________________________________________")




class run_the_bootstraps:
    def __init__(self,x,y,z,deg,lamd): #x,y and z have to be the column vector where each element corresponds
        self.lamd =lamd
        self.x, self.y, self.z =  x ,y ,z
        self.boot_error_list_training = []
        self.nr_bootstraps = 10
        self.beta_list = []
        self.deg = deg
        self.run_bootstrap_on_training_data()
        #self.plotio()



    def run_bootstrap_on_training_data(self):
        for k in range(self.nr_bootstraps):
            BOOT = bootstrap(len(self.x))
            x_train = BOOT.generate_training_data(self.x)
            y_train = BOOT.generate_training_data(self.y)
            z_train = BOOT.generate_training_data(self.z)
            B = Ridge_(x_train,y_train,z_train,self.deg,self.lamd)
            self.boot_error_list_training.append(B.errors())
            self.beta_list.append(B.beta)
        self.boot_error_list_training = np.array(self.boot_error_list_training)
    #    hist = np.histogram(self.boot_error_list_training)

    def plotio(self):
        plt.hist(self.boot_error_list_training[:,0])
        plt.show()

    def run_bootstrap_on_test_data(self):
        for k in range(self.nr_bootstraps):
            BOOT = bootstrap(len(self.x))
            x_test = BOOT.generate_test_data(self.x)
            y_test = BOOT.generate_test_data(self.y)
            z_test = BOOT.generate_test_data(self.z)
            B = Ridge_(x_test,y_test,z_test,self.deg,self.lamb)
            self.boot_error_list_test.append(B.errors())
        self.boot_error_list_test = np.array(self.boot_error_list_test)
    #    hist = np.histogram(self.boot_error_list_training)
        #plt.hist(self.boot_error_list_test[:,0])
        #plt.show()

class var_and_bias:
    def __init__(self,X,z,beta_list): #beta list has all the betas for all the fits from different bootstraps,
        self.beta_list = beta_list
        self.z = z
        self.X = X
        self.the_fits = self.generate_a_list_of_fits()
        self.average_fit = self.generate_average_of_fits()
        self.bias = self.generate_the_bias()
        self.var = self.generate_the_variance()

    def generate_a_list_of_fits(self):
        listio = []
        for beta in self.beta_list:
            fit = self.X.dot(beta)
            listio.append(fit)
        return listio

    def generate_average_of_fits(self):
        sums =0
        nr=0
        for fit in self.the_fits:
            sums=sums+fit
            nr=nr+1
        if nr>0:
            sums =sums/nr
        return sums
    def generate_the_bias(self):
        bias = (self.z-self.average_fit)**2
        return sum(bias)

    def generate_the_variance(self):
        sums =0
        nr=0
        for fit in self.the_fits:
            sums=sums + (fit-self.average_fit)**2
            nr=nr+1
        if nr>0:
            sums =sums/nr
        return sum(sums)


#x=np.array([1,2,3,4,5,6,7,8,9,10])
#y=np.array([1.3,1.4,1.5,1.4,1.3,1.2,1.5,1.6,1.5,1.2,])
#plt.plot(x,y)
#plt.show()

#A = OLS_main() #here we prepare all the variables we need
#Dd = run_the_bootstraps(A.x_,A.y_,A.z_) #here we feed the variables from A instance to the bootstrap class
