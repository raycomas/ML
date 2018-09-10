function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z.
%

temp = sigmoid(z);
g = temp.*(1 - temp);

endfunction
