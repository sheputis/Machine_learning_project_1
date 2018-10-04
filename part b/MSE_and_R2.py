from ML_classes_ridge import *
degrees = range(1,6)
lamda = 2
MSE_list=[]
R2_list=[]
for i in degrees:
    A = Ridge_main(i,lamda)
    errors = A.errors()
    MSE_list.append(errors[0])
    R2_list.append(errors[1])
plt.figure(1)
plt.plot(degrees,MSE_list,'ro')
plt.xlabel('degrees',fontsize=20)
plt.ylabel('MSE',fontsize=20)
plt.title('MSE with noise of variance 0.01',fontsize = 25)
plt.figure(2)
plt.plot(degrees,R2_list,'ro')
plt.xlabel('degrees',fontsize=20)
plt.ylabel('R2 score',fontsize=20)
plt.title('R2 score with noise of variance 0.01', fontsize=25)
plt.show()

print("MSE")
print(MSE_list)
print("R2 score")
print(R2_list)
