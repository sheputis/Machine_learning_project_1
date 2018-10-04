from ML_classes_ridge import *
for i in range(5):
    A = Ridge_main(i+1,0.1)
    A.variance_in_beta()
