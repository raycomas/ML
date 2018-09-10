function [weights, cost] = lrNumeric(X, y, lambda = 0, maxIter = 400)
%LRNUMERIC Numerically computes the weights for linear regression 
%   LRNUMERIC(X,y, lambda, maxIter) Numerically computes the weights for linear regression 
%   using the minimize() optimization function.
%       X is the matrix whose rows are the data points of the training set
%       y is the vector of labels, where y(n) corresponds to X(n,:)
%       lambda is the regularization constant to use. If this is zero,
%              regularization is not used
%       maxIter is the maximum number of iterations to perform
%  Returns:
%    weights the computed weight vector, which contain an extra w0 bias component
%    cost    an array of cost values at each iteration, which can be examined or
%            plotted to determine whether (and how quickly) convergence occurred

N = length(y);   % number of training examples
X = [ ones(N, 1) X ];   % Append bias x0 = 1 to each data point

weights = zeros(size(X,2), 1);   % initialize weights to all zeros
[weights cost iterations] = minimize(weights, @lrCost, maxIter, X, y, lambda);

end
