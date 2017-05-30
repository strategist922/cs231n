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
  for i in range(X.shape[0]):
      out = np.dot(X[i, :], W)
    #   out -= np.max(out)
      out = np.exp(out)
      out = out / (out.sum())
      label = y[i]
      loss += -np.log(out[label])
      for j in range(W.shape[1]):
          if j == label:
              dW[:, j] += (out[j] - 1) * X[i, :]
          else:
              dW[:, j] += out[j] * X[i, :]
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg * W
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
  out = np.dot(X, W)
  out = np.exp(out)
  out_sum = np.sum(out, axis=1)
  out_sum = out_sum.reshape((out_sum.shape[0], 1))
  out = out / out_sum
  minus_out = np.zeros(out.shape)
  minus_out[np.arange(y.shape[0]), y] = 1
  dW = np.dot(X.T, (out-minus_out)) / X.shape[0]
  out_choice = out[np.arange(y.shape[0]), y]
  out_choice = -np.log(out_choice)
  loss = np.sum(out_choice) / X.shape[0] + 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
