function y = lgPredict(weights, X)
%lgPredict Make a prediction for the value of data points
%  lgPredict(weights, X) 
%  Make a prediction for the value of data points using logistic regression
%    weights  the weight vector obtained from logistic regression
%    X        a matrix of data points, expressed as a row vectors. 
%  Returns
%    y  the predicted values for X
%
% Copyright (C) 2018 Ray Comas
%

  if (size(X,2) != size(weights))
    X = [ones(size(X,1), 1) X];
  endif

  y = sigmoid(X * weights);

endfunction
