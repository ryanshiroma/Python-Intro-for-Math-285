
####Importing Packages ##################################################
import numpy as np

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
import timeit



####### python basics ################


# variables

# create scalar x.
# NOTE: semicolon is not needed for any line of code to supress output
x=1

#Vectors and Matrices
# NOTE: python is case sensitive
x=np.array([[1,2,3]]) # 1x3 row vector
y=np.array([[1],[2],[3]]) # 3x1 column vector
X=np.array([[4,5,6],[7,8,9]]) # 2x3 matrix
square_mat=np.array([[1,2,3],[4,5,6],[7,8,9]]) # 2x3 matrix)
allones= np.ones((50,50)) # create matrix of all ones - equivalent to " ones(50,50) " in matlab

x.shape #The dimensions of x is (1,3) - this is like the " size() " function in matlab
X.shape #The dimensions of X is (2,3)


# Accessing vector or matrix elements
# NOTE: the first element starts at ZERO! Also, square brackets are used instead of parentheses.
x[0]  # the first element of x
X[0,:] # first row of X
X[:,0] # first column of X
X[:,0:1] # first two columns of X
X[1,1] # X_2,2 element
x[x>1] # all elements not equal to 6
np.where(x>1) # indices of x where x>1. equivalent to " find() " in matlab


# matrix manipulation and linear algebra
np.transpose(X) #transpose of X - equivalent to " X' " in matlab
np.dot(np.transpose(X),X) # X'X - equivalent to " X'X " in matlab
X*X # element wise multiplication equivalent to " X.*X " in matlab
X+1 # add 1 to all elements in X - equivalent to " X+ones(3,2) " in matlab
X+x # add the row x to all rows in X - equivalent to " X+repmat(x,3,1) " in matlab
np.diag(x[0,:]) #create a diagonal matrix with the elements of x - equivalent to " diag(x) " in matlab
np.linalg.inv(square_mat) # get the inverse of X - equivalent to " inv(X) " in matlab
lambdas, V = np.linalg.eig(square_mat) # eigen values and vectors - equivalent to " [lambdas,V]=eig(square_mat) "


## for loops, if statements, and functions #########


count =0
for i in range(10):
    count=count+1
print(count)


if count==5:
    print("the count is 5")
elif count <5:
    print("the count is less than 5")
else:
    print("the count is greater than 5")



def myfunc(x,y): # a simple function that returns the sum of its parameters
    return x+y

myfunc(4,6)


## loading data #######################################################
trainimages=np.load("trainimages.npy")
testimages=np.load("testimages.npy")
trainlabels=np.load("trainlabels.npy")
testlabels=np.load("testlabels.npy")



#### Plotting ###########################################################

plt.figure() #plotting a digit
plt.imshow(trainimages[0].reshape(28,28),cmap=plt.gray())


plt.figure() # simple plot
x=range(1,21)
y=np.random.poisson(20,20)
plt.plot(x,y)
plt.xlabel("day")
plt.ylabel("# of times I hear the word 'hella'")
plt.yticks(range(max(y)+2))
plt.xticks(x)
plt.title("Hellas per day")
plt.grid()




########################## KNN and PCA #######################################################


### Run KNN just once with k=3 #######################################################
begin=timeit.default_timer()
mdl= neighbors.KNeighborsClassifier(3,weights="uniform",n_jobs=2)
mdl.fit(trainimages[:10000], trainlabels[:10000])
pred=mdl.predict(testimages[:1000])
score=1-accuracy_score(testlabels[:1000],pred)
print(score)
print(str(timeit.default_timer()-begin)+ " seconds")





############################### PCA #######################################################


###### SVD function method ###### THIS MIGHT TAKE TOO LONG TO RUN!!

X_tilde=trainimages-np.mean(trainimages,axis=0)

U,S,V=np.linalg.svd(X_tilde)
U=np.load("U.npy")
S=np.load("S.npy")
V=np.load("V.npy")

S=np.diag(S)

## how many dimensions do I need to keep to contain 95% of the variance?
cum_s = np.cumsum(S*S)
cum_s = cum_s/cum_s[-1]
s = np.argmax(cum_s>0.95)

Y = np.dot(X_tilde,V[:,:s]) ## reduced dimension matrix
Xtest_tilde=testimages-np.mean(trainimages,axis=0)
Ytest = np.dot(Xtest_tilde,V[:,s])




##### PCA function method ##### USE THIS INSTEAD OF SVD
begin=timeit.default_timer()
pca = PCA()
pca.fit(trainimages)
S=pca.explained_variance_ratio_
cum_s = np.cumsum(S)
cum_s = cum_s/cum_s[-1]
s = np.argmax(cum_s>0.95)
Y=pca.transform(trainimages)[:,:s]
Ytest=pca.transform(testimages)[:,:s]
print(str(timeit.default_timer()-begin)+ " seconds")


##Run KNN once with k=3 on reduced dataset
begin=timeit.default_timer()
mdl2= neighbors.KNeighborsClassifier(3,weights="uniform",n_jobs=2)
mdl2.fit(Y,trainlabels)
pred=mdl2.predict(Ytest)
score=1-accuracy_score(testlabels,pred)
print(score)
print(str(timeit.default_timer()-begin)+ " seconds")







######################################################################################
################   MORE STUFF!!!   ###################################################
######################################################################################


############################# custom functions #####################################


###custom knn ##############################
from myfunctions import knn
k=3
test_labels=knn(k,trainimages[:10000],trainlabels[:10000],testimages[:1000])
score=1-accuracy_score(testlabels[:1000],test_labels[k-1])
print(score)

##repeat on reduced set
from myfunctions import knn
k=3
begin=timeit.default_timer()
test_labels=knn(3,Y[:10000],trainlabels[:10000],Ytest[:1000])
score=1-accuracy_score(testlabels[:1000],pred)
print(score)
print(str(timeit.default_timer()-begin)+ " seconds")






###custom local kmeans ##############################
from myfunctions import localkmeans
k=3
test_labels=localkmeans(k,trainimages[:10000],trainlabels[:10000],testimages[:1000])
score=1-accuracy_score(testlabels[:1000],test_labels[k-1])
print(score)


##repeat on reduced set
from myfunctions import localkmeans
k=3
test_labels=localkmeans(k,Y[:10000],trainlabels[:10000],Ytest[:1000])
score=1-accuracy_score(testlabels[:1000],test_labels[k-1])
print(score)





############################ KFolds ############################

##### KFOLD and KNN on full data set
from myfunctions import linear_weights
n=trainimages.shape[0]
n=1000
k=6
kf = KFold(n=n, n_folds=6, shuffle=True)
error=np.zeros((k,1))
score=0

start = timeit.default_timer()
for k in range(1,k+1):
    for train_index, test_index  in kf:
        #WEIGHTING OPTIONS:linear_weights,'uniform', 'distance'
        classifier= neighbors.KNeighborsClassifier(k,weights="uniform",n_jobs=2)
        classifier.fit(trainimages[train_index], trainlabels[train_index])
        pred=classifier.predict(trainimages[test_index])
        score+=accuracy_score(trainlabels[test_index],pred)
    score/=6
    print("k="+str(k) + " Accuracy score:" +str(score))
    error[k-1]=score
    score=0
print error
stop = timeit.default_timer()
print str(stop-start) +" seconds"

plt.plot(range(1,k+1), error)
plt.title("Classification Rate")
plt.xlabel("k-neighbors")
plt.ylabel("percent correct")
plt.show()





#### KFold and KNN on reduced dimension data set
from myfunctions import linear_weights
n=Y.shape[0]
n=2000
k=6
kf = KFold(n=n, n_folds=6, shuffle=True)
error=np.zeros((k,1))
score=0

start = timeit.default_timer()
for k in range(1,k+1):
    for train_index, test_index  in kf:
        #WEIGHTING OPTIONS:linear_weights,'uniform', 'distance'
        classifier= neighbors.KNeighborsClassifier(k,weights="uniform",n_jobs=2)
        classifier.fit(Y[train_index], trainlabels[train_index])
        pred=classifier.predict(Y[test_index])
        score+=accuracy_score(trainlabels[test_index],pred)
    score/=6
    print("k="+str(k) + " Accuracy score:" +str(score))
    error[k-1]=score
    score=0
print error

stop = timeit.default_timer()
print str(stop-start) +" seconds"

plt.plot(range(1,k+1), error)
plt.title("Classification Rate")
plt.xlabel("k-neighbors")
plt.ylabel("percent correct")
plt.show()


