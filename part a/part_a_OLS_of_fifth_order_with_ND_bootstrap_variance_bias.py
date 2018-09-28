from ML_classes import *
bias=[]
variance=[]
for deg in range(9):

    print("running bootstrap of degree %s " % (deg+1))
    A = OLS_main(deg+1)
    B = run_the_bootstraps(A.x_,A.y_,A.z_,deg+1)
    C = var_and_bias(A.X_,A.z_,B.beta_list)
    print('bias')
    print(C.bias)
    print('variance')
    print(C.var)
    bias.append(C.bias)
    variance.append(C.var)
bias=np.array(bias)
variance=np.array(variance)
plt.plot(bias,label ='bias')
plt.plot(variance, label ='variance')
plt.plot(variance+bias, label ='sum')
plt.legend()
plt.show()
