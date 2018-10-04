from ML_classes import *

degree = 5
A = OLS_main(degree) #here we prepare all the variables we need
bootstraps = run_the_bootstraps(A.x_,A.y_,A.z_,degree) #here we feed the variables from A instance to the bootstrap class
bootstraps.run_bootstrap_on_test_data()
bootstraps.plot_test_errors()
