from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, num_dim = X.shape
    num_classes = W.shape[1]

    # loss = WX_i + softmax(WX_i) + reg * sum(W * W)
    for i in xrange(num_train) :
        scores = X[i].dot(W)
        scores = scores - max(scores) # shift the score to -max(scores), for numeric stability reason
        correct_class_score = scores[y[i]]
        sigma = np.sum(np.exp(scores))
        loss += -correct_class_score + np.log(sigma)    
        for j in xrange(num_classes) :
            temp = np.exp(scores[j]) / sigma
            if (j == y[i]) :
                dW[: , j] +=  (temp - 1) * X[i] 
            else :
                dW[: , j] += temp * X[i]
    loss /= num_train
    loss += reg * np.sum(W * W) / 2
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train, num_dim = X.shape
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    scores -= np.max(scores, axis = 1).reshape(-1, 1)
    currect_class_score = scores[xrange(num_train), y]
    sigma = np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
    loss -= np.sum(currect_class_score)
    loss += np.sum(np.log(sigma))
    loss /= num_train
    loss += reg * np.sum(W * W) / 2
    # print(loss.shape)
    temp = np.exp(scores) / sigma
    temp[xrange(num_train), y] -= 1
    dW = (X.T).dot(temp)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
