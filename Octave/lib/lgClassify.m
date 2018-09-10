function y = lgClassify(K, weights, X)
%LGCLASSIFY Classify data points
%   LGCLASSIFY(weights, X) Classify a set of data points using 
%   weights computed with logistic regression
%       K       A vector contaiing the classes
%       weights is the matrix whose column vectors are the weights 
%               from logistic regression for K: weights(:,n) are the weights
%               of the classifier for K(n)
%       X is a matrix of data points, expressed as a row vectors
%   K and weights can be obtained as the output of lgTrain.
%   Returns a vector indicating the classification of the data points in X,
%   so y(n) is the classification of X(n,:)
%

[~, y] = max(lgPredict(weights, X), [], 2);
y = K(y);

endfunction
