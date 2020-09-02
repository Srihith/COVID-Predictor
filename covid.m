%clear old data
clc;
clear;

%load data
load("input.mat");
load("output.mat");

%set testing data and training data
input_training = input([1:45],:);
output_training = output([1:45],:);
input_testing = input([46:52],:);
output_testing = output([46:52],:);

%train the model
hiddenLayerSize = 1000;
net1 = patternnet(hiddenLayerSize);
net1.divideParam.trainRatio = 70/100;
net1.divideParam.valRatio   = 15/100;
net1.divideParam.testRatio  = 15/100;
[net1,tr] = train(net1,input_training',output_training');
out = net1(input_testing')';

%output data
save('output.mat','output');

