function p = nnPredict(weights, layer_sizes, X)
%NNPREDICT Make a prediction for the value of data points
%   NNPREDICT(weights, layer_sizes, X) Make a prediction for the value of data points
%   using a neural network
%       weights is the weight vector obtained from nnTrainClassifier
%       X is a matrix of data points, expressed as a row vectors. 
%   Returns the predicted values for X
%

N = size(X, 1);
theta_base = 0;
layers = length(layer_sizes);

for layer = 1:(layers - 1)
  a = layer_sizes(layer);
  b = layer_sizes(layer + 1);
  theta = reshape(weights(theta_base + (1:b*(a+1))), b, a+1);
  theta_base = theta_base + b*(a+1);
  X = sigmoid([ones(N, 1) X] * theta');
endfor

p = X;

endfunction
