import numpy as np
from random import shuffle

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
  y_pred = np.matmul(X, W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  ############################################################################
  r_w = np.sum(np.square(W))
  for i in range(y_pred.shape[0]):
      y_act = np.zeros((y_pred.shape[1]))
      y_act[y[i]-1] = 1
      y_act = y_act.flatten()
      unnorm = y_pred[i] - np.max(y_pred[i]) # Stable softmax
      h_i = np.exp(unnorm[y[i]-1])/np.sum(np.exp(unnorm))
      l_i = h_i + (reg * r_w)
      loss += l_i
      
      print 'input=> ', (X[np.newaxis, i]).T.shape
      print "exp=> ",np.exp(unnorm), ' shape=> ', np.exp(unnorm).shape, (np.exp(unnorm)[:, np.newaxis]).shape
      print "y=> ",y_act, ' shape=> ', y_act.shape, (y_act[:, np.newaxis]).shape
      print "diff=> ", ((np.exp(unnorm)[:, np.newaxis]) - (y_act[:, np.newaxis])).flatten(), ' shape=> ', ((np.exp(unnorm)[:, np.newaxis]) - (y_act[:, np.newaxis])).shape
      dW += np.matmul(X[np.newaxis, i], (((np.exp(unnorm)[:, np.newaxis]) - (y_act[:, np.newaxis])).flatten()).T)
      
      
  loss = loss/y_pred.shape[0]
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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

