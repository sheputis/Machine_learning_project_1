from ML_classes import *

#the confidence interval in Beta depends only on the confidence interval in y
#meaning that the degree og polynomial that we fit does not matter
A = OLS_main(5) #creating an instance of a fitting of degree 5
A.variance_in_beta() #writing out diagonal variancce elements
A.plot_everything() #plotting the fit
