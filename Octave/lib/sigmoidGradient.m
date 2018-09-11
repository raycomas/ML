function g = sigmoidGradient(z)
%sigmoidGradient returns the gradient of the sigmoid function at z
%  g = sigmoidGradient(z) 
%  computes the gradient of the sigmoid function evaluated at z.
%
% Copyright (C) 2018 Ray Comas
%

  temp = sigmoid(z);
  g = temp.*(1 - temp);

endfunction
