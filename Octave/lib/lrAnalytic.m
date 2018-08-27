function [weights] = lrAnalytic(X, y)
%LRANALYTIC Computes the analytic solution for linear regression 
%   LRANALYTIC(X,y) computes the analytic solution for linear 
%   regression using the normal equations.
%       X is the matrix whose rows are the data points of the training set
%       y is the vector of labels, where y(n) corresponds to X(n,:)
%   Returns the computed weight vector, which contain an extra w0 bias component

N = length(y);   % number of training examples
X = [ ones(N, 1) X ];   % Append bias x0 = 1 to each data point

weights = pinv(X' * X) * X' * y;

end
