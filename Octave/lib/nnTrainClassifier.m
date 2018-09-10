function [weights, layer_sizes, classes] = nnTrainClassifier(X, labels, layer_sizes, lambda = 0, maxIters = 400)
%nnTrainClassifier Train a neural network classifier
%  [weights layer_sizes classes] = nnTrainClassifier(X, labels, layer_sizes, lambda, maxIter) 
%  Train a neural network classifier
%    X       a matrix of data points, expressed as a row vectors
%    labels  the corresponding labels for X: labels(n) is the label for X(n,:)
%    layer_sizes   a vector of hidden layer sizes
%    lambda  the regularization constant to use. If this is zero,
%            regularization is not used
%    maxIter the maximum number of iterations to perform
%  Returns
%    layer_sizes  a vector containing the sizes of the network
%    weights      the computed weights for the neural network
%    classes      a vector of the unique values in labels.
% 

  classes = unique(labels)(:);
  layer_sizes = [size(X,2) layer_sizes length(classes)];
  num_weights = (1 + layer_sizes)(1:end-1) * layer_sizes(2:end)';

  weights = (rand(num_weights,1) * 0.12 - 0.12);
  X = [ones(size(X, 1), 1) X];

  [weights ~ ~] = minimize(weights, @nnClassifierCost, maxIters, ...
                           classes, X, labels, layer_sizes, lambda);
endfunction
