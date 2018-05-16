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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N,D = X.shape
  C = W.shape[1]
  for i in range(N):
    score_i = X[i,:].dot(W)
    score_i -= np.max(score_i)
    denominator = np.sum(np.exp(score_i), axis=0)
    p_i = np.exp(score_i)/denominator
        
    loss += -score_i[y[i]] + np.log(denominator)
    
    for j in range(C):
        dW[:,j] += p_i[j] * X[i,:]
        if j==y[i]:
            dW[:,j] -= X[i,:]
  loss = loss/N + reg*np.sum(W*W)
  dW = dW/N + 2*reg*W
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
  N,D = X.shape
  C = W.shape[1]
  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[:,None]
  p = np.exp(scores)/np.sum(np.exp(scores),axis=1)[:,None]
  loss = np.sum(-np.log(p[np.arange(N), y]))/N + reg*np.sum(W*W)
  incorrect = np.zeros_like(p)
  incorrect[np.arange(N), y] = 1
  dW = X.T.dot((p-incorrect))/N + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

