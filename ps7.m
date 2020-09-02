%Question 1
clear;


randomSelect = randperm(length(X));
randomImages = X(randomSelect(1:16),:);

for i=1:size(randomImages,1)
    figure(1);
    subplot(4,4,i);
    imshow(reshape(randomImages(i,:),[20,20]), []);
end

p = predict(Theta1, Theta2, X);
accuracy = sum(~(y - p))/length(y);

z = [-10,0,10];
sigmoidGradient(z);

nn_params = [Theta1(:) ; Theta2(:)];
lambda = 0;
input_layer_size = size(X,2);
hidden_layer_size = size(Theta1,1);
num_labels = 10;
nnCostFunction(nn_params, input_layer_size,hidden_layer_size,10,X, y, lambda);
lambda = 1;
nnCostFunction(nn_params, input_layer_size,hidden_layer_size,10,X, y, lambda);

lambda = 0;
[J grad] = nnCostFunction(nn_params, input_layer_size,hidden_layer_size,10,X, y, lambda);
checkNNGradients(lambda);

lambda = 3;
[J grad] = nnCostFunction(nn_params, input_layer_size,hidden_layer_size,10,X, y, lambda);
checkNNGradients(lambda);

iters = [50,100,200,400];
lambdas = [0,1,2,4];
for i = 1:4
    for j = 1:4
    % Initialize parameters (randInitializeWeights.m)
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    % Train the network
    options = optimset('MaxIter', iters(i));
    lambda = lambdas(j);
    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer_size, num_labels, X, y, lambda);
    % Now, costFunction is a function that takes in only one argument (the % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
    p = predict(Theta1, Theta2, X);
    accuracy(j,i) = sum(~(y - p))/length(y);
    end
end

accuracy