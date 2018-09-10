function y = nnClassify(K, weights, layer_sizes, X)
%NNCLASSIFY Predict the label of an input given a trained neural network
%    y = NNCLASSIFY(K, weights, layer_sizes, X) Classify a set of data points using 
%    a neural network
%       K        A vector contaiing the classes
%       weights  A vector containing the weights of the trained network
%       layer_sizes  A vector containing the size of the layers
%                    layer_sizes(1) is the input layer,
%                    layer_sizes(end) is the output layer,
%                    The other elements are the hidden layers
%                    This should be the same layer_sizes parameter used to
%                    train the network
%       X is a matrix of data points, expressed as a row vectors
%   K and weights can be obtained as the output of nnTrainClassifier.
%   Returns a vector indicating the classification of the data points in X,
%   so y(n) is the classification of X(n,:)
%

[~, y] = max(nnPredict(weights, layer_sizes, X), [], 2);
y = K(y);

endfunction
