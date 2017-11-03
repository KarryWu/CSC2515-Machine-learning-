'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import math

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(0,10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        means[i,:] = i_digits.mean(axis = 0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for i in range(0,10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        cov = i_digits - i_digits.mean(axis = 0)
        for j in range(0,64):
            for k in range(0,64):
                covariances[i,j,k] = np.dot(cov[:,j],cov[:,k].T)/len(i_digits) 
        covariances[i,:,:] = covariances[i,:,:] + 0.01*np.eye(64)  ##add 0.01I on diagnal
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    image_data = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...
        image_data_i = np.log(cov_diag).reshape((8,8))
        image_data.append(image_data_i)
        
    all_image_data = np.concatenate(image_data, 1)
    plt.imshow(all_image_data, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    gene_like = np.zeros((digits.shape[0],10))
    for n in range(digits.shape[0]):
        for k in range(0,10):
            gene_like[n,k] = (2*math.pi)**(-64/2)*np.linalg.det(covariances[k])**(-1/2)*exp(-1/2*np.mat((digits[n]-means[k]).reshape((64,1)).T)*np.mat(np.linalg.inv(covariances[k]))*np.mat((digits[n]-means[k]).reshape((64,1))))      
    return np.log(gene_like)

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gene_like1 = np.exp(generative_likelihood(digits, means, covariances))
    sum = np.zeros((digits.shape[0],1))
    for n in range(digits.shape[0]):
        for k in range(0,10):
            sum[n] = sum[n] + 0.1 * gene_like1[n,k]
            
    con_like = np.zeros((digits.shape[0],10))
    for n in range(digits.shape[0]):
        for k in range(0,10):
            con_like[n,k] = 0.1*gene_like1[n,k]/sum[n]
    return np.log(con_like)

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    sum_cond_likelihood = 0
    for i in range(len(labels)):
        sum_cond_likelihood += cond_likelihood[i,int(labels[i])]
    ave_cond_likelihood = sum_cond_likelihood/len(labels)
    return ave_cond_likelihood

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pred = zeros((len(digits),1))
    for i in range(len(digits)):
        pred[i] = argmax(cond_likelihood[i])
    
    return pred

def calculate_accuracy(digits, labels, predictions):
    correct = 0 
    for i in range(len(digits)):
        if predictions[i] == labels[i]:
            correct += 1
    
    return correct/len(digits)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    plot_cov_diagonal(covariances)

    # Evaluation
    avg_codlike_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print("The average conditional likelihood of train data is",avg_codlike_train)
    
    avg_codlike_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("The average conditional likelihood of test data is",avg_codlike_test)
    
    pred_train = classify_data(train_data, means, covariances)
    pred_test = classify_data(test_data, means, covariances)
    
    accuracy_train = calculate_accuracy(train_data, train_labels, pred_train)
    print("The accuracy of the model on train data is",accuracy_train)
    accuracy_test = calculate_accuracy(test_data, test_labels, pred_test)
    print("The accuracy of the model on test data is",accuracy_test)
        

if __name__ == '__main__':
    main()
    
