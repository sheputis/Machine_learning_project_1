from ML_classes_ridge import *
degree= range(1,11)
lambda_list=[0]
for lamd in lambda_list:
    bias=[]
    variance=[]
    for deg in degree:
#    for deg in [1,5,10,15]:
        print("running bootstrap of degree %s and lamda %s" % ((deg),lamd))
        A = Ridge_main(deg,lamd)
        B = run_the_bootstraps(A.x_,A.y_,A.z_,deg,lamd)
        C = var_and_bias(A.X_,A.z_,B.beta_list)
    #    print('bias')
    #    print(C.bias)
    #    print('variance')
    #    print(C.var)
        bias.append(C.bias)
        variance.append(C.var)
    bias=np.array(bias)
    variance=np.array(variance)
#    print(variance)
    plt.plot(degree,bias,label ='bias' )
    plt.plot(degree,variance, label ='variance' )
    plt.plot(degree,variance+bias, label ='variance + bias' )


plt.xlabel('degree',fontsize = 20)
plt.ylabel('the mean error',fontsize = 20)
plt.title('the plot of the total mean error of OLS',fontsize = 25)
plt.legend()
plt.show()
#print(variance)
