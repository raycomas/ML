function showData(X, y, predicted_y)
%SSHOWDATA Display 2D data in a grid
%   SHOWDATA(X, y) displays 2D data
%   stored in X in a grid, corresponding labels are in y,
%   optional predictions in predicted_y. Highlight the image
%   if its predicted label doesn't match its actual label 

% If predicted labels are not provided, set predicted_y to be the actual labels
if (!exist('predicted_y', 'var') || isempty(predicted_y))
  predicted_y = y;
  show_predicted = 0;
else
  show_predicted = 1;
endif

[rows cols] = size(X);

% Number of images on each grid side - display area must be square
display_side = sqrt(rows);

% Size of each image side in pixels - image must be square
image_side = sqrt(cols);

% Size in pixels of each side of the display grid
grid_side = display_side * image_side;

% Create blank display
display_grid = zeros(grid_side, grid_side);

% Copy each image into a grid square in the display grid
curr_img = 1;
for i = 1:display_side
	for j = 1:display_side
		
    image = reshape(X(curr_img, :), image_side, image_side)';
    if (y(curr_img) != predicted_y(curr_img)) image = 1 - image; endif
		display_grid((j - 1) * image_side + (1:image_side), ...
		             (i - 1) * image_side + (1:image_side)) = image;
		curr_img = curr_img + 1;
	endfor
endfor

% show labels
reshape(y, display_side, display_side)

% show predicted labels if they were provided
if (show_predicted == 1)
   reshape(predicted_y, display_side, display_side) 
endif

% Display image grid
colormap(gray);
h = imagesc(display_grid, [0 1]);
axis image off;
drawnow;

end
