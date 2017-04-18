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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    f_i = X[i].dot(W)
    f_i -= np.max(f_i) # shift values to make the largest to be 0, a trick to avoid numerical instability
    correct_class_score = f_i[y[i]]

    sum_i = np.sum(np.exp(f_i))
    loss += -np.log(np.exp(correct_class_score) / sum_i)

    for j in range(num_classes):
      p_j = np.exp(f_i[j]) / sum_i
      dW[:, j] += (p_j - (y[i] == j)) * X[i]

  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

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
  num_train = X.shape[0]

  s = X.dot(W)
  s -= np.max(s, axis=1, keepdims=True) # avoid numerical instability
  sum_s = np.sum(np.exp(s), axis=1, keepdims=True)

  p = np.exp(s) / sum_s
  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  index = np.zeros_like(p)
  index[np.arange(num_train), y] = 1
  Q = p - index
  dW = X.T.dot(Q)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW

