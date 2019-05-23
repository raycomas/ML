clear;
addpath('./lib');

% load dataset
load handwriting.mat;

% Train NN classifier, using 80 nodes in one hidden layer
% This will take a while ...
[weights layer_sizes classes] = nnTrainClassifier(training_data, training_labels, [ 80 ]);

% See how well we did on the training data
training_predictions = nnClassify(classes, weights, layer_sizes, training_data);
100 * sum(training_predictions == training_labels)/length(training_predictions)
%Should be around 98%-99% accurate

% And on the test data (which wasn't used for training)
test_predictions = nnClassify(classes, weights, layer_sizes, test_data);
100 * sum(test_predictions == test_labels)/length(test_predictions)
% Should be around 94%-95% accurate

% Try it with regularization, and see if that helps any ...
[weights layer_sizes classes] = nnTrainClassifier(training_data, training_labels, [ 80 ], 0.1);

% Try the test data again ...
test_predictions = nnClassify(classes, weights, layer_sizes, test_data);
100 * sum(test_predictions == test_labels)/length(test_predictions)
% Should be a little better, 97%-98% accurate

% Now let's try the validation set
val_predictions = nnClassify(classes, weights, layer_sizes, validation_data);
100 * sum(val_predictions == validation_labels)/length(val_predictions)
