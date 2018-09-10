function [cost, grad] = nnClassifierCost(weights, K, X, y, layer_sizes, lambda = 0)
%NNCLASSIFIERCOST Compute cost and gradient for neural networks, intended for
%       use with fminunc(), minimize(), and similar optimization functions.
%   [cost, grad] = NNCLASSIFIERCOST(weights, X, y, layer_sizes, lambda)
%   computes the cost and gradient for neural networks.
%       weights is the linear regression weight vector
%       K is the vector of possible classes
%       X is the matrix whose rows are the data points of the training set
%         each row should already contain a bias term
%       y is the vector of labels, where y(n) corresponds to X(n,:)
%       layer_sizes is a vector of layer sizes, where
%                   layer_sizes(1) is the size of the input layer,
%                   layer_sizes(end) is the size of the output layer,
%                   the intermediate elements are the sizes of the hidden layer(s)
%       lambda is the regularization constant to use. If this is zero,
%              regularization is not used
%

N = length(y);   % number of training examples

theta_base = 0;
layers = length(layer_sizes);
reg_weights = [];

% Feed forward
A{1} = X;
for layer = 1:(layers - 1)
  a = layer_sizes(layer);
  b = layer_sizes(layer + 1);

  theta = reshape(weights(theta_base + (1:b*(a+1))), b, a+1);
  T{layer} = theta;
  theta_base = theta_base + b*(a+1);

  reg_weights = [ reg_weights ; theta(:,2:end)(:) ]; 

  z = A{layer} * theta';
  Z{layer + 1} = z;
  A{layer + 1} = [ ones(size(z), 1) sigmoid(z) ];    
endfor

A_n = A{layers}(:,2:end);
y_vec = K(:)' == y;
cost = (lambda*sumsq(reg_weights)/2 ...
		      - sum(sum(y_vec.*log(A_n) + (1 - y_vec).*log(1 - A_n))))/N;

% Backpropagation
grad = [];
D = A_n - y_vec;
for layer = (layers - 1):-1:1 
  T_n = T{layer};
  A_n = A{layer};  
  curr_grad = (D'*A_n + lambda*[zeros(size(T_n,1),1) T_n(:,2:end)])/N;  
  grad = [ curr_grad(:) ; grad ];
  
  if (layer > 1)
	  D = (D*T_n)(:,2:end).*sigmoidGradient(Z{layer});
  endif
endfor

endfunction
