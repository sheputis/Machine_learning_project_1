from ML_classes import *



lambda_list=[0.01,0.1,0.5,2,6,20]
for lamd in lambda_list:
    bias=[]
    variance=[]
    for deg in range(10):
        print("running bootstrap of degree %.2f and lamda %.2f" % ((deg+1),lamd))
        A = Lasso_main(deg+1,lamd)
        B = run_the_bootstraps(A.x_,A.y_,A.z_,deg+1,lamd)
        C = var_and_bias(A.X_,A.z_,B.beta_list)
    #    print('bias')
    #    print(C.bias)
    #    print('variance')
    #    print(C.var)
        bias.append(C.bias)
        variance.append(C.var)
    bias=np.array(bias)
    variance=np.array(variance)
    print('uo')
    print(bias.shape)
    plt.plot(bias,label ='bias %d' % lamd)
    #plt.plot(variance, label ='variance')
    #plt.plot(variance+bias, label ='lamda %.2f' % lamd)



plt.legend()
plt.show()
print(bias)
print(variance)
