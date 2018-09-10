function g = sigmoid(z)
%sigmoid Compute sigmoid (logistic) function
%  g = sigmoid(z) 
%  computes the sigmoid of z.
%

  g = 1.0 ./ (1.0 + exp(-z));

endfunction
