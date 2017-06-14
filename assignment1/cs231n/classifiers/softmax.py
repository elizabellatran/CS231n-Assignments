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
  
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #loop computs softmax and loss gradient 
  for i in xrange(num_train):

    #vector scores 
    scores = X[i].dot(W) # prediction of training sample 
    scores -= np.max(scores)

    # calcu probabilities of samples --probability of the correct class /
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    
    # softmax loss
    loss += -np.log(probabilities[y[i]])
    #probabilities[y[i]] = -1
    #for j in xrange(num_classes):
    #  dW[:,j] += X[i] * probabilities[j]
    
    
    #comput gradient of inner sum given i 
    # dW is adjusted by each row being the X[i] by probab vector
    dW[:,y[i]] -= X[i] 
    for j in xrange(num_classes):
      dW[:,j] += X[i] * probabilities[j]

    #  if j == y[i]:
    #    dW[:,j] += (X[i] * (probabilities[j] - 1))
    #  else:
    #    dW[:,j] += X[i] * probabilities[j]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  #compute regularization 
  loss += reg * np.sum(W * W)
  dW += reg*W


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

  num_train = X.shape[0] 

  scores = X.dot(W) # prediction of training sample  = NxD * DxC = NxC 
  scores -= np.max(scores, axis=1, keepdims=True) # max of every sample for norm for stability  -- http://cs231n.github.io/linear-classify/  
  sum_scores = np.sum(np.exp(scores), axis=1, keepdims=True)

  probabilities  = np.exp(scores)/sum_scores  #probabs of samples  
  loss = np.sum(-np.log(probabilities [np.arange(num_train), y]))  # L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}    

  ind = np.zeros_like(probabilities )
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(probabilities - ind)   # DxN * NxC = DxC


  #comput gradient of inner sum given i 
  # dW is adjusted by each row being the X[i] by probab vector
  #for j in xrange(num_classes):
  #  dW[:,j] += X[i,:] * probabilities[j] # DxN * 
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg*W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

