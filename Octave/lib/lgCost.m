function [cost, grad] = lgCost(weights, X, y, lambda = 0)
%LGCOST Compute cost and gradient for logistic regression, intended for
%       use with fminunc(), minimize(), and similar optimization functions.
%   [cost, grad] = LGCOST(X, y, weights, lambda) computes the cost and 
%   gradient for logistic regression. 
%       weights is the linear regression weight vector
%       X is the matrix whose rows are the data points of the training set
%         each row should already contain a bias term
%       y is the vector of labels, where y(n) corresponds to X(n,:)
%       lambda is the regularization constant to use. If this is zero,
%              regularization is not used
%  Returns:
%      cost - the computed cost for the given weights
%      grad - the gradient for the given weights

N = length(y);   % number of training examples
if (size(X,2) != size(weights))
  X = [ones(size(X,1), 1) X];
endif

temp = sigmoid(X*weights);

cost = (lambda*sumsq(weights(2:end))/2 - sum(y.*log(temp) + (1-y).*log(1-temp)))/N;
grad = 1/N * ((temp - y)' * X)' + [0; lambda/N * weights(2:end)];

end
