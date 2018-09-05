function y = lgPredict(weights, X)
%LGPREDICT Make a prediction for the value of data points
%   LRPREDICT(weights, X) Make a prediction for the value of data points
%   using logistic regression
%       weights is the weight vector obtained from logistic regression
%       X is a matrix of data points, expressed as a row vectors. 
%   Returns the predicted values for X

y = sigmoid([ones(size(X,1),1) X] * weights);

end