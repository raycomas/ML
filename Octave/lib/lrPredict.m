function y = lrPredict(weights, X)
%lrPredict Make a prediction for the value of data points
%  lrPredict(weights, X) 
%  Make a prediction for the value of data points regression using 
%  the normal equations.
%    weights  is the weight vector obtained from linear regression
%    X        is a matrix of data points, expressed as a row vectors. 
%  Returns
%    y  the predicted values for X
%
% Copyright (C) 2018 Ray Comas
%

  if (size(X,2) != size(weights))
    X = [ones(size(X,1), 1) X];
  endif

  y = X * weights;

endfunction
