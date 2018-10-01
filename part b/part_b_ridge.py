from ML_classes import *

lambda_list=[0]#,0.1,10]#,0.5,2,6,20]
for lamd in lambda_list:
    bias=[]
    variance=[]
    for deg in range(6):
#    for deg in [1,5,10,15]:
        print("running bootstrap of degree %s and lamda %s" % ((deg+1),lamd))
        A = Ridge_main(deg+1,lamd)
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
#    print(variance)
#    plt.plot(bias,label ='bias %.2f' % lamd)
    #plt.plot(variance, label ='variance %.2f' % lamd)
    plt.plot(variance+bias, label ='lamda %.2f' % lamd)



plt.legend()
plt.show()
#print(variance)
