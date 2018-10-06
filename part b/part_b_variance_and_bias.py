from ML_classes_ridge import *
degree= range(1,13)
lambda_list=[0,0.1,0.5,2,6,20]
for lamd in lambda_list:
    bias=[]
    variance=[]
    for deg in degree:
        print("running bootstrap of degree %s and lamda %s" % ((deg),lamd))
        A = Ridge_main(deg,lamd)
        B = run_the_bootstraps(A.x_,A.y_,A.z_,deg,lamd)
        C = var_and_bias(A.X_,A.z_,B.beta_list)
        bias.append(C.bias)
        variance.append(C.var)
    bias=np.array(bias)
    variance=np.array(variance)
#    print(variance)
    #plt.plot(degree,bias,label ='bias %.2f' % lamd)
    #plt.plot(degree,variance, label ='variance %.2f' % lamd)
    plt.plot(degree,variance+bias, label ='lamda %.2f' % lamd)


plt.xlabel('degree',fontsize = 20)
plt.ylabel('the mean error',fontsize = 20)
plt.title('the plot of the total mean error(Variance + bias) of ridge fits for different lambdas',fontsize = 25)
plt.legend()
plt.show()
#print(variance)
