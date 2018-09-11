function [cost, grad] = lrCost(weights, X, y, lambda = 0)
%lrCost Compute cost and gradient for linear regression, intended for
%       use with fminunc(), minimize(), and similar optimization functions.
%  [cost grad] = lrCost(weights, X, y, lambda) 
%  computes the cost and gradient for linear regression. 
%    weights  the current linear regression weight vector
%    X        the matrix whose rows are the data points of the training set
%             a bias term will be added if not already present
%    y        the vector of labels, where y(n) corresponds to X(n,:)
%    lambda   the regularization constant to use. If this is zero,
%             regularization is not used
%  Returns
%     cost  the computed cost for the given weights
%     grad  the gradient for the given weights
%
% Copyright (C) 2018 Ray Comas
%

  N = length(y);   % number of training examples
  if (size(X,2) != size(weights))
    X = [ones(size(X,1), 1) X];
  endif

  temp = X*weights - y;

  cost = 1/(2*N) * (sumsq(temp) + lambda*sumsq(weights(2:end)));
  grad = 1/N * (temp' * X)' + [0; lambda/N * weights(2:end)];

endfunction
