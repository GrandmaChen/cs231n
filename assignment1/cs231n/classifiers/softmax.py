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

    # W.shape = (3073, 10)
    # X.shape = (500, 3073)

    num_classes = W.shape[1]  # 10
    num_train = X.shape[0]  # 500

    for i in xrange(num_train):

        # Scores of current image
        scores = X[i].dot(W)

        # Normalization trick to avoid numerical instability
        # http://cs231n.github.io/linear-classify/#softmax
        scores -= scores.max()

        correct_class_score = scores[y[i]]

        # Sum up loss over all examples
        loss += -np.log(np.exp(correct_class_score) / np.exp(scores).sum())

        # Get the gradient of W
        # https://blog.csdn.net/pjia_1008/article/details/66972060
        # Gradient of all correct items
        dW[:, y[i]] += ((np.exp(correct_class_score) /
                         np.exp(scores).sum()) - 1) * X[i]

        # Gradient of each incorrect item
        for j in xrange(num_classes):
            if j != y[i]:
                dW[:, j] += (np.exp(scores[j]) /
                             np.exp(scores).sum()) * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # np.set_printoptions(threshold=np.nan)

    num_classes = W.shape[1]  # 10
    num_train = X.shape[0]  # 500

    # Loss
    scores_all = X.dot(W)  # shape = (500, 10)
    scores_all -= scores_all.max()

    scores_correct = scores_all[range(num_train), y]  # shape = (500,)

    # sum(axis=1) means the sum of all items (e to the power of Sj)
    # within this image
    # shape(500,)
    margin_all = -np.log(np.exp(scores_correct) /
                         np.exp(scores_all).sum(axis=1))

    loss = margin_all.mean() + reg * np.sum(W * W)

    # Gradient
    # axis=1 to keep rows
    sums_of_each_row = np.exp(scores_all).sum(axis=1, keepdims=True)

    factors = np.exp(scores_all) / sums_of_each_row
    factors[range(num_train), y] -= 1

    # Multiply
    dW = (X.T).dot(factors)
    dW /= num_train
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
