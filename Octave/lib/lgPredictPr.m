function p = lgPredictPr(weights, X)
%lgPredictPr  Determine the probability that the point x is in each class
%  p = lgPredictPr(weights, X)
%  Determine the probability that the point X is in each class
%    weights  the weight vector obtained from logistic regression
%    X        a matrix of data points, expressed as a row vectors. 
%  Returns
%    y  a matrix where y(n,k) is the probablity that x(n,:) belongs to class k,
%       and where sum(y(n,:)) = 1.
%
% Copyright (C) 2018 Ray Comas
%

  if (size(X,2) != size(weights))
    X = [ones(size(X,1), 1) X];
  endif

  w_1 = weights(:,2:end);  
  p_1 = 1.0 ./ (1 + sum(exp(X * w_1), 2));
  p = [ p_1 p_1.*exp(X * w_1)];
  
endfunction
