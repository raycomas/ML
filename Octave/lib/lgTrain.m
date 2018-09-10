function [weights, classes] = lgTrain(X, labels, lambda = 0, maxIters = 400)
%lgTrain Train a set of one-vs-many classifiers
%  [weights classes] = lgTrain(X, labels, lambda, maxIters) 
%  Train a set of one-vs-many classifiers using logistic regression
%    X       a matrix of data points, expressed as a row vectors
%    labels  the corresponding labels for X: labels(n) is the label for X(n,:)
%    lambda  the regularization constant to use. If this is zero,
%            regularization is not used
%    maxIter the maximum number of iterations to perform during optimization
%  Returns
%    weights a matrix whose column vecords are the weights of the classifiers,
%            so weights(:,n) is the classifier for K(n). This matrix may be 
%            passed as a parameter to lgClassify to classify data points
%    classes a vector of the unique values in labels. Pass this vector as
%            a parameter to lgClassify
% 

  classes = unique(labels)(:);
  weights = [];
  for i=classes'
    weights = [ weights  lgNumeric(X, labels == i, lambda, maxIters)];
  endfor

endfunction
