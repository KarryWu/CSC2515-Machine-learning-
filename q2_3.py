'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    for i in range(0,10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        new_line1 = np.zeros((1,64))
        new_line2 = np.ones((1,64))
        new_idigits = np.vstack((i_digits,new_line1,new_line2))
        for j in range(0,64):
            eta[i,j] = np.sum(new_idigits[:,j])/len(new_idigits)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    plot_data = []
    for i in range(10):
        img_i = class_images[i]
        # ...
        img_i_data = img_i.reshape((8,8))
        plot_data.append(img_i_data)
        
    all_image_data = np.concatenate(plot_data, 1)
    plt.imshow(all_image_data, cmap='gray')
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    generated_data = binarize_data(eta)
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    gene_like = np.ones((bin_digits.shape[0],10))
    for n in range(bin_digits.shape[0]):
        for k in range(0,10):
            for j in range(0,64):
                gene_like[n,k] = gene_like[n,k] * eta[k,j]**bin_digits[n,j]*(1-eta[k,j])**(1-bin_digits[n,j])     
    return np.log(gene_like)
   

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gene_like1 = np.exp(generative_likelihood(bin_digits, eta))
            
    con_like = np.zeros((bin_digits.shape[0],10))
    for n in range(bin_digits.shape[0]):
        for k in range(0,10):
            con_like[n,k] = 0.1*gene_like1[n,k]
    return np.log(con_like)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    sum_cond_likelihood = 0
    for i in range(len(labels)):
        sum_cond_likelihood += cond_likelihood[i,int(labels[i])]
    ave_cond_likelihood = sum_cond_likelihood/len(labels)
    return ave_cond_likelihood

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    pred = zeros((len(bin_digits),1))
    for i in range(len(bin_digits)):
        pred[i] = argmax(cond_likelihood[i])
    return pred

def calculate_accuracy(bin_digits, labels, predictions):
    correct = 0 
    for i in range(len(bin_digits)):
        if predictions[i] == labels[i]:
            correct += 1
    
    return correct/len(bin_digits)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)
    
    avg_codlike_train = avg_conditional_likelihood(train_data, train_labels, eta)
    print("The average conditional likelihood of train data is",avg_codlike_train)
    
    avg_codlike_test = avg_conditional_likelihood(test_data, test_labels, eta)
    print("The average conditional likelihood of test data is",avg_codlike_test)
    
    pred_train = classify_data(train_data, eta)
    pred_test = classify_data(test_data, eta)
    
    accuracy_train = calculate_accuracy(train_data, train_labels, pred_train)
    print("The accuracy of the model on train data is",accuracy_train)
    accuracy_test = calculate_accuracy(test_data, test_labels, pred_test)
    print("The accuracy of the model on test data is",accuracy_test)

if __name__ == '__main__':
    main()
