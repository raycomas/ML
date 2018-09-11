function g = tanhGradient(z)
%tanhGradient returns the gradient of the tanh function at z
%  g = tanhGradient(z) 
%  computes the gradient of the tanh function evaluated at z.
%
% Copyright (C) 2018 Ray Comas
%

  temp = tanh(z);
  g = 1 - temp.*temp;

endfunction
