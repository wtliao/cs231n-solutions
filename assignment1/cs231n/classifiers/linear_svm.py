import numpy as np
from random import shuffle
import pdb

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  margin_mask = np.zeros((num_train, num_classes))
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        margin_mask[i,j] = 1

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  for i in range(num_train):
    for j in range(num_classes):
      if  margin_mask[i,j] == 1:
        if j != y[i]:
          dW[:,j] += X[i]
          dW[:,y[i]] -= X[i]
      else:
        continue
  dW /= num_train
  dW += 2*reg*W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  margins = scores - scores[range(num_train),y].reshape((num_train,-1))+1
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins = np.maximum(0,margins)
  margins[np.arange(num_train),y] = 0
  loss += np.sum(margins)
  loss /= num_train
  loss += reg*np.sum(W*W)

  margin_masks = (margins>0).astype(np.float)
  incorret_counts = np.sum(margin_masks,1)
  margin_masks[np.arange(num_train),y] = -incorret_counts
  dW = X.T.dot(margin_masks)
  # I dont know why it doesnot work
  # dW[:,y] -= (X*incorret_counts.reshape((num_train,1))).T
  # pdb.set_trace()
  # samples,j = np.where(margin_masks)
  # dW[:,j] += X[samples]
  
  dW /= num_train
  dW += reg*W*2
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
