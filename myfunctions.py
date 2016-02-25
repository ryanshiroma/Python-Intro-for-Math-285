import numpy as np
import numpy.matlib as ml
from scipy.spatial import distance
from scipy import stats
import random
import timeit
import matplotlib.pyplot  as plt
from sklearn.metrics import accuracy_score


###define linear weighting custom function
def linear_weights(x):
        k=x.shape[1]
        x=x.transpose()
        weights=((x[k-1,:]-x)/(x[k-1,:]-x[0,:])).transpose()
        return weights


class Fold(object): # My own "Fold" object. This object contains the indices for training and validation
    def __init__(self):
        self.train = None
        self.val = None

    def train(self):
        return self.train

    def val(self):
        return self.val


# k-nearest neighbor function


def knn(k,train,train_lab, test,t_lab=None, d_meas="euclidean",weights="equal"):
    """
Parameters
----------
k : largest k to be computed
train : numpy array containing the training data
train_lab : list or numpy array containing the labels for the training data
test : numpy array containing the test data
d_meas : default distance measure is euclidean, also available are 'cityblock', 'cosine', etc. see the numpy 'cdist' documentation for more options
weights : "equal" for equal weighting, "linear" for linearly decreasing weights
t_lab : OPTIONAL PARAMETER useful for debugging
"""
    test_size = len(test) # number of test observations
    test_lab = np.empty([k,test_size],dtype=int)
    begin = timeit.default_timer()
    uclasses=np.unique(train_lab)
    for i in range(test_size): #loop through all test observations
        distances = distance.cdist(train,[test[i]],d_meas) # calculate distances from test sample to all training data
        classes = np.argpartition(distances,tuple(range(k)),axis=None)
        #temp=train[classes[:len(k)-1]].reshape(28*len(classes[:len(k)-1]),28)

        for l in range(k):
            if weights=="linear": #linear weighting
                d_n1=distances[classes[l+1]][0] #k+1 nearest neighbor
                d_1=distances[classes[0]][0] # nearest neighbor
                wdist=(d_n1-distances[classes[:l+1]])/(d_n1-d_1) #compute weights
                neighbors=train_lab[classes[:l]] #get labels of nearest k neighbors
                weightsums=np.zeros([len(uclasses)],dtype=float) #initialization
                for a in np.unique(neighbors): #loop through the nearest k neighbors
                    weightsums[a]=sum(wdist[neighbors==a])  #sum up weights for each class
                test_lab[l, i]=np.argmax(weightsums) #set test label to the class with the highest weight
            elif weights=="equal": #majority voting
                test_lab[l, i] = stats.mode(train_lab[classes[:l+1]])[0]
        if i%10==0:
            stop = timeit.default_timer()
            print "index: " + str(i) + "/"+ str(test_size) +"  percent complete: "+ str((i*100)/test_size) +"%  estimated time remaining: " + str((stop-begin)*(test_size-i)/(i+1)) + " seconds"
    print "total time: " + str(stop-begin) + " seconds"
    return test_lab




def localkmeans(k,train,train_lab, test,d_meas="euclidean"):
    """
    k : largest k to be computed
    train : numpy array containing the training data
    train_lab : list or numpy array containing the labels for the training data
    test : numpy array containing the test data
    d_meas : default distance measure is euclidean, also available are 'cityblock', 'cosine', etc. see the numpy 'cdist' documentation for more options
    :return:
    """
    test_size = len(test)
    test_lab = np.empty([k,test_size])#,dtype=int)
    classes=np.unique(train_lab)
    localknn_images = np.zeros((k*len(classes),train.shape[1]),dtype=float)
    begin = timeit.default_timer()

    for i in range(test_size):

        distances = distance.cdist(train,[test[i]],d_meas)
        for j in classes:

            class_index=np.where(train_lab==j)
            class_distances=distances[class_index]
            class_images=train[class_index]
            sorted_indices = np.argpartition(class_distances,tuple(range(k)),axis=None)[:k]
            localknn_images[range(j,k*len(classes),len(classes))]=class_images[sorted_indices]

        iter_means= np.zeros((len(classes),train.shape[1]),dtype=float)

        for l in range(k):
            iter_means+=localknn_images[l*len(classes):(l+1)*len(classes)]
            kmeans_distances = distance.cdist(iter_means,[test[i]*(l+1)],d_meas)
            test_lab[l,i] = np.argmin(kmeans_distances)

        if i%10==0:
            stop = timeit.default_timer()
            print "index: " + str(i) + "/"+ str(test_size) +"  percent complete: "+ str((i*100)/test_size) +"%  estimated time remaining: " + str((stop-begin)*(test_size-i)/(i+1)) + " seconds"
    print "total time: " + str(stop-begin) + " seconds"
    return test_lab


def find_class_means(train,train_lab,d_meas="euclidean"):
    classes=np.unique(train_lab)
    print(classes)
    class_means = np.empty([len(classes),train.shape[1]],dtype=float)
    for i in range(len(classes)):
        class_means[i, :] = np.mean(train[train_lab==classes[i]],axis=0)
    return class_means



def compute_error(true_labels, pred_labels):
    return sum(true_labels == pred_labels)/float(len(true_labels))


def kfolds(k, train, randomize=False):
    n = len(train)
    folds = [Fold() for _ in range(n)]

    if randomize:
        indices = random.shuffle(range(n))
    else:
        indices = range(n)

    for i in range(k):

        folds[i].val = np.array(indices[int(i*np.floor(n/k)):int((i+1)*np.floor(n/k))+1]).astype(int)
        folds[i].train = np.hstack((indices[:int(i*np.floor(n/k))],indices[int((i+1)*np.floor(n/k))+1:])).astype(int)

    return folds



