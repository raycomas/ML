function [weights, K] = lgTrain(X, labels, lambda = 0, maxIters = 400)
%LGRAIN Train a set of one-vs-many classifiers
%   LGTRAIN(X, labels, lambda, maxIters) Train a set of one-vs-many classifiers
%   using logistic regression
%     K      is a vector containing the possible values of hte labels
%     X      is a matrix of data points, expressed as a row vectors
%     labels is the corresponding labels for X: labels(n) is the label for X(n,:)
%   Returns
%     K       a vector of the unique values in labels. Pass this vector as
%             a parameter to lgClassify
%     Weights a matrix whose column vecords are the weights of the classifiers,
%             so weights(:,n) is the classifier for K(n). This matrix may be 
%             passed as a parameter to lgClassify to classify data points
% 

  K = unique(labels)(:);
  weights = [];
  for i=K'
    weights = [ weights  lgNumeric(X, labels == i, lambda, maxIters)];
  endfor

endfunction
