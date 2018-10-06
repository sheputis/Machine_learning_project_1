import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ML_classes_real_data import *


# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
terrain1 = terrain1[:500,:500]
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

degree= range(1,10)
lambda_list=[0.5,2,6,20]
for lamd in lambda_list:
    bias=[]
    variance=[]
    for deg in degree:
        print("running bootstrap of degree %.2f and lamda %.2f" % ((deg),lamd))
        A = Ridge_main(deg,lamd, terrain1)
        B = run_the_bootstraps(A.x_,A.y_,A.z_,deg,lamd)
        C = var_and_bias(A.X_,A.z_,B.beta_list)
        bias.append(C.bias)
        variance.append(C.var)
    bias=np.array(bias)
    variance=np.array(variance)
    plt.plot(degree,bias,label ='bias of lambda %.2f' % lamd)
    plt.plot(degree,variance, label ='variance of lambda %.2f' % lamd)
    plt.plot(degree,variance+bias, label ='var+bias of lamda %.2f' % lamd)


plt.xlabel('degree',fontsize = 20)
plt.ylabel('the mean error',fontsize = 20)
plt.title('the plot of variance, bias of ridge fits for different lambdas ',fontsize = 25)
plt.legend()
plt.show()
