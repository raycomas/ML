function [weights, layer_sizes] = nnTrainRegression(X, labels, layer_sizes, lambda = 0, maxIters = 400)
%nnTrainRegression Train a regression neural network
%  [weights layer_sizes ] = nnTrainRegression(X, labels, layer_sizes, lambda, maxIter) 
%  Train a regression neural network
%    X       a matrix of data points, expressed as a row vectors
%    labels  the corresponding labels for X: labels(n) is the label for X(n,:)
%    layer_sizes   a vector of hidden layer sizes
%    lambda  the regularization constant to use. If this is zero,
%            regularization is not used
%    maxIter the maximum number of iterations to perform
%  Returns
%    layer_sizes  a vector containing the sizes of the network
%    weights      the computed weights for the neural network
%
% Copyright (C) 2018 Ray Comas
%

  layer_sizes = [size(X,2) layer_sizes 1];
  num_weights = (1 + layer_sizes)(1:end-1) * layer_sizes(2:end)';

  weights = (rand(num_weights,1) * 0.12 - 0.12);
  X = [ones(size(X, 1), 1) X];

  [weights, ~, ~] = minimize(weights, @nnRegressionCost, maxIters, ...
                               X, labels, layer_sizes, lambda);
endfunction
