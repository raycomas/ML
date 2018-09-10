function y = nnClassify(classes, weights, layer_sizes, X)
%nnClassify Predict the label of an input given a trained neural network
%  y = nnClassify(K, weights, layer_sizes, X) 
%  Classify a set of data points using a neural network
%    classes      a vector contaiing the classes
%    weights      a vector containing the weights of the trained network
%    layer_sizes  a vector containing the size of the layers
%                   layer_sizes(1) is the input layer,
%                   layer_sizes(end) is the output layer,
%                   the other elements are the hidden layers
%    X            a matrix of data points, expressed as a row vectors
%  K and weights can be obtained as the output of nnTrainClassifier.
%  Returns
%    y   a vector indicating the classification of the data points in X,
%        so y(n) is the classification of X(n,:)
%

  [~, y] = max(nnPredict(weights, layer_sizes, X), [], 2);
  y = classes(y);

endfunction
