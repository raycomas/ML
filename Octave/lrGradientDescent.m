function [weights, cost] = lrGradientDescent(X, y, alpha, lambda, maxIter)
%LRGRADIENTDESCENT Performs gradient descent for linear regression
%   theta = LRGRADIENTDESCENT(x, y, alpha, maxIter) performs gradient
%   descent for linear regression
%       X is the matrix whose rows are the data points of the training set
%       y is the vector of labels, where y(n) corresponds to X(n,:)
%       alpha is the learning rate to useful
%       lambda is the regularization constant to use. If this is zero,
%              regularization is not used
%       maxIter is the maximum number of iterations to perform
%  Returns:
%      weights - the computed weight vector, which contain an extra w0 bias component
%      cost - an array of cost values at each iteration, which can be examined or
%             plotted to determine whether (and how quickly) convergence occurred
addpath('./lib');
  
N = length(y);   % number of training examples
X = [ ones(N, 1) X ];   % append bias x0 = 1 to each data point
threshold = 0.000001;   % threshold for convergence

cost = [];   % cost at each iteration
weights = zeros(size(X,2), 1);   % initialize weights to all zeros
for iter = 1:maxIter
    [curr_cost grad] = lrCost(weights, X, y, lambda);
    cost = [ cost ; curr_cost ];
	  old_weights = weights;
    weights = weights - alpha*grad;
    if norm(weights - old_weights) < threshold, break; endif;
end

end
