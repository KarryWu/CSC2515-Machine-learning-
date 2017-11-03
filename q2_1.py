'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        dist = self.l2_distance(test_point)      ##get the distances
        indexes = argsort(dist)   ##get the indexes in increasing order of distances

        tie = 1
        while tie == 1:
            index = indexes[0:k]      ##get the first K indexes
            labels = self.train_labels[index]   
            labels = labels.astype(np.int64)    ##change the type
            ##get the mode
            count = np.bincount(labels)
            tie = 0
            for i in range(len(count)-1):
                for j in range(i+1,len(count)):
                    if count[i] == count[j] and count[i] == np.max(count):   ##tie happened
                        tie = 1
                        k = k - 1
                        break 
                if tie == 1:
                    break
            if tie == 0:   
                break
                    
        digit = np.where(count == np.max(count))
        return digit[0]
       

def cross_validation(knn, k_range=np.arange(1,15)):
    accuracy_table = []
    for k in k_range:
        print(k)  ##test point.....
        kf = KFold(n_splits = 10)
        sum_accuracy = 0
        for train_index, test_index in kf.split(knn.train_data):
            train_input, test_input = knn.train_data[train_index] , knn.train_data[test_index]
            train_output, test_output = knn.train_labels[train_index] , knn.train_labels[test_index]
            knn1 = KNearestNeighbor(train_input, train_output)
            accuracy = classification_accuracy(knn1,k,test_input, test_output)
            sum_accuracy = sum_accuracy + accuracy
            mean_accuracy = sum_accuracy/10
        accuracy_table.append(mean_accuracy)     ##build a table of accuracies of different k
    optimal_k = argmax(accuracy_table) + 1   ##value of k = index of k + 1
    print("optimal_k =", optimal_k)
    print("average accuracy cross folds of optimal k =", max(accuracy_table))
    return optimal_k
        

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    number_correct = 0
    for i in range(len(eval_data)):
        if knn.query_knn(eval_data[i],k) == eval_labels[i]:
            number_correct += 1
    
    return number_correct/len(eval_data)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    predicted_label = knn.query_knn(test_data[0], 1)
    print(predicted_label)
    
    ## Accuracy calculation:
    accuracy_train_k1 = classification_accuracy(knn,1,train_data, train_labels)
    print("The accuracy of training data when k = 1 is", accuracy_train_k1)
    accuracy_train_k15 = classification_accuracy(knn,15,train_data, train_labels)
    print("The accuracy of training data when k = 15 is", accuracy_train_k15)
    accuracy_test_k1 = classification_accuracy(knn,1,test_data, test_labels)
    print("The accuracy of test data when k = 1 is", accuracy_test_k1)
    accuracy_test_k15 = classification_accuracy(knn,15,test_data, test_labels)
    print("The accuracy of test data when k = 15 is", accuracy_test_k15)
    
    ## Accuracy under the optimal K
    opt_k = cross_validation(knn, k_range=np.arange(1,16))
    accuracy_train_optk = classification_accuracy(knn,opt_k,train_data, train_labels)
    print("training accuracy of optimal K =", accuracy_train_optk)
    accuracy_test_optk = classification_accuracy(knn,opt_k,test_data, test_labels)
    print("test accuracy of optimal K =", accuracy_test_optk)
    
    
    
if __name__ == '__main__':
    main()