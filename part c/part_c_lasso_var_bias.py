from ML_classes_lasso import *

degree= range(1,10)
lambda_list=[0.01,0.1,0.5,2,6,20]
for lamd in lambda_list:
    bias=[]
    variance=[]
    for deg in degree:
        print("running bootstrap of degree %.2f and lamda %.2f" % ((deg),lamd))
        A = Lasso_main(deg,lamd)
        B = run_the_bootstraps(A.x_,A.y_,A.z_,deg,lamd)
        C = var_and_bias(A.X_,A.z_,B.beta_list)
        bias.append(C.bias)
        variance.append(C.var)
    bias=np.array(bias)
    variance=np.array(variance)
    plt.plot(degree,bias,label ='bias of lambda %.2f' % lamd)
    plt.plot(degree,variance, label ='variance of lambda %.2f' % lamd)
    #plt.plot(degree,variance+bias, label ='var+bias of lamda %.2f' % lamd)


plt.xlabel('degree',fontsize = 20)
plt.ylabel('the mean error',fontsize = 20)
plt.title('the plot of variance, bias of lasso fits for different lambdas to investigate their scales',fontsize = 25)
plt.legend()
plt.show()
