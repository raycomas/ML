function showSample(X, y, row_size, predicted_y)

  rand_ix = randperm(size(X,1), row_size*row_size);  

  if (!exist('predicted_y', 'var') || isempty(predicted_y)) 
   showData(X(rand_ix,:), y(rand_ix));
  else
   showData(X(rand_ix,:), y(rand_ix), predicted_y(rand_ix));   
  endif
  
endfunction
