function [Z W S] = pcaReduce(X, ret_var = 0.99)
%pcaReduce  Perform Principal Component Analysis on X
%  [Z W S] = pcaReduce(X, ret_var)
%  Performs Principal Component Analysis on X and reduce its dimensionality
%    X        the data to analyze. Each column corresponds to one set of
%             observations. The data should alreedy be normalized so that
%             all the values have similar order of magnitude
%    ret_var  the desired retained variance (default 0.99)
%  Returns
%    Z      the trasform of X 
%    W      the matrix that was used to transform X (by computing X * W)
%    S      The variance percentages for each column of W
%
% Copyright (C) 2018 Ray Comas
%

  [U S ~] = svd(cov(X));
  s_diag = diag(S);
  var_tot = sum(s_diag);
  
  k = 1;
  var_p = s_diag(k);
  while var_p/var_tot < ret_var
    k = k +1;
    var_p = var_p + s_diag(k);
  endwhile

  W = U(:,1:k);
  Z = X * W;
  S = s_diag(1:k) ./ var_tot;
  
endfunction
