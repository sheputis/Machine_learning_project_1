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


degrees = range(1,6)
lamda = 0.1
MSE_list=[]
R2_list=[]
for i in degrees:
    A = Ridge_main(i,lamda,terrain1)
    errors = A.errors()
    MSE_list.append(errors[0])
    R2_list.append(errors[1])
plt.figure(1)
plt.plot(degrees,MSE_list,'ro')
plt.xlabel('degrees',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.title('MSE ',fontsize = 25)
plt.figure(2)
plt.plot(degrees,R2_list,'ro')
plt.xlabel('degrees',fontsize=20)
plt.ylabel('R2 score',fontsize=20)
plt.title('R2 score', fontsize=25)
plt.show()

print("MSE")
print(MSE_list)
print("R2 score")
print(R2_list)
