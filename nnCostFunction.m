function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


m = size(X,1);
yNew = zeros(m,num_labels);
for i=1:num_labels
    yNew(:,i) = (y == i);
end
X = [ones(size(X,1),1), X];
a1 = [ones(size((X*Theta1'),1),1), sigmoid(X*Theta1')];
p = sigmoid(a1*Theta2');
J = (1/m)*sum(sum((-yNew.*log(p)) - ((1-yNew).*log(1-p)))) + (lambda/(2*m)).*((sum(sum(Theta1(:, 2:end).^2)))+(sum(sum(Theta2(:, 2:end).^2))));

delta3 = p-yNew;
delta2 = delta3*Theta2(:, 2:end).*sigmoidGradient(X*Theta1');
Theta1_grad = Theta1_grad+(1/m)*(X'*delta2)'
Theta2_grad = Theta2_grad+(1/m)*(a1'*delta3)'

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+(lambda/m).*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+(lambda/m).*Theta2(:,2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
