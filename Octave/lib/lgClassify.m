function y = lgClassify(classes, weights, X)
%lgClassify Classify data points
%  y = lgClassify(classes, weights, X) 
%  Classify a set of data points using  weights computed with logistic regression
%    classes  a vector contaiing the classes
%    weights  the matrix whose column vectors are the weights 
%             from logistic regression for classes: weights(:,n) are the weights
%             of the classifier for classes(n)
%    X        a matrix of data points, expressed as a row vectors
%  classes and weights can be obtained as the output of lgTrain.
%  Returns 
%    y  a vector indicating the classification of the data points in X,
%       so y(n) is the classification of X(n,:)
%
% Copyright (C) 2018 Ray Comas
%

  [~, y] = max(lgPredict(weights, X), [], 2);
  y = classes(y);

endfunction
