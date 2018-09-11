function [weights] = lrAnalytic(X, y, lambda = 0)
%lrAnalytic Computes the analytic solution for linear regression 
%  weights = lrAnalytic(X, y, lambda) 
%  computes the analytic solution for linear regression using the normal equations.
%    X       is the matrix whose rows are the data points of the training set
%    y       is the vector of labels, where y(n) corresponds to X(n,:)
%    lambda  is the regularization constant to use. If this is zero,
%            regularization is not used
%  Returns 
%    weights  the computed weight vector, which contain an extra w0 bias component
%
% Copyright (C) 2018 Ray Comas
%

  N = length(y);   % number of training examples
  X = [ ones(N, 1) X ];   % Append bias x0 = 1 to each data point

  weights = pinv(X'*X + lambda*eye(size(X,2))) * X' * y;

endfunction
