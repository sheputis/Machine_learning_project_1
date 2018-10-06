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


for i in range(5):
    A = Ridge_main(i+1,0.1,terrain1)
    A.variance_in_beta()
