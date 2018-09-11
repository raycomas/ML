function [weights, cost] = lgNumeric(X, y, lambda = 0, maxIter = 400)
%lgNumeric Numerically computes the weights for logistic regression 
%  [weights cost] = lgNumeric(X,y, lambda, maxIter) 
%  Numerically computes the weights for logistic regression using 
%  the minimize() optimization function.
%    X       the matrix whose rows are the data points of the training set
%    y       the vector of labels, where y(n) corresponds to X(n,:)
%    lambda  the regularization constant to use. If this is zero,
%            regularization is not used
%    maxIter the maximum number of iterations to perform during optimization
%  Returns
%    weights the computed weight vector, which contain an extra w0 bias component
%    cost    an array of cost values at each iteration, which can be examined or
%            plotted to determine whether (and how quickly) convergence occurred
%
% Copyright (C) 2018 Ray Comas
%

  N = length(y);   % number of training examples
  X = [ ones(N, 1) X ];   % Append bias x0 = 1 to each data point

  weights = zeros(size(X,2), 1);   % initialize weights to all zeros
  [weights cost ~] = minimize(weights, @lgCost, maxIter, X, y, lambda);

endfunction
