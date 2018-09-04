addpath('./lib');

% Load data set
load house-prices.mat;

% sample a couple of records
sample_ix = randperm(1000,10);
[data(sample_ix,:) price(sample_ix)]

% radomly select 150 records as the validation set
val_index = randperm(1000,150);
val_data = data(val_index,:);
val_price = price(val_index);
%remove validation records from the data set
data(val_index,:) = [];
price(val_index) = [];

% randomly select 150 records as the test set
test_index = randperm(850,150);
test_data = data(test_index,:);
test_price = price(test_index);
%remove test records from the data set
data(test_index,:) = [];
price(test_index) = [];

% normalize the training data
mu_data = mean(data);
sigma_data = std(data);
mu_price = mean(price);
sigma_price = std(price);

training_data = (data - mu_data) ./ sigma_data;
training_price = (price - mu_price) ./ sigma_price;

% create some models
% data1 is just the normalized data
data1 = training_data;
% data2 = data with squared terms
data2 = [data1 data1(:,1).^2 data1(:,2).^2 data1(:,3).^2];
% data3 = data2 with cross terms
data3 = [data2 data1(:,1).*data1(:,2) ...
               data1(:,1).*data1(:,3) ...
               data1(:,2).*data1(:,3)];

% compute the weights for hte 3 models
weights1 = lrAnalytic(data1, training_price);
weights2 = lrAnalytic(data2, training_price);
weights3 = lrAnalytic(data3, training_price);

% normalize the validation set using the same normalization values
% used for the training set
cross_val_data = (val_data - mu_data) ./ sigma_data;
cross_val_price = (val_price - mu_price) ./ sigma_price;

% create corresponding validation sets for each models
val_data1 = cross_val_data;
val_data2 = [val_data1, val_data1(:,1).^2 ...
                        val_data1(:,2).^2 ...
                        val_data1(:,3).^2];
val_data3 = [val_data2 val_data1(:,1).*val_data1(:,2) ...
                       val_data1(:,1).*val_data1(:,3) ...
                       val_data1(:,2).*val_data1(:,3)];

% compute costs
cost1 = lrCost(weights1, val_data1, cross_val_price)
cost2 = lrCost(weights2, val_data2, cross_val_price)
cost3 = lrCost(weights3, val_data3, cross_val_price)

% let's try model #2 - normalize the test data
norm_test_data = (test_data - mu_data) ./ sigma_data;

% create the proper data set for model 2
test_data2 = [norm_test_data ...
                norm_test_data(:,1).^2 ...
                norm_test_data(:,2).^2 ...
                norm_test_data(:,3).^2];

% select 10 records at random to predict
sample_index = randperm(150, 10);
sample_data2 = test_data2(sample_index,:);
predicted_prices = lrPredict(weights2, sample_data2);

% predicted_prices are normalized values; convert back to dollar amounts
predicted_prices = (predicted_prices * sigma_price) + mu_price;

% show the result
[test_data(sample_index,:) predicted_prices test_price(sample_index,:) ...
  100 * abs(1 - test_price(sample_index,:) ./ predicted_prices)]
