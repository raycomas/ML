function y = nnPredict(weights, layer_sizes, X)
%nnPredict Make a prediction for the value of data points
%  nnPredict(weights, layer_sizes, X) 
%  Make a prediction for the value of data points using a neural network
%    weights      the weight vector obtained from nnTrainClassifier
%    layer_sizes  a vector containing the sizes of the network layers
%                   layer_sizes(1) is the size of the input layer
%                   layer_sizes(end) is the size of the output layer
%                   the other layers are the sizes of the hidden layers
%    X            a matrix of data points, expressed as a row vectors. 
%  Returns
%    y  A vector containing the predictions of the points in X:
%       y(n) is the prediction of X(n,:)
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

  y = X;

endfunction
