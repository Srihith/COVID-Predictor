function p = predict (Theta1, Theta2, X)
    X = [ones(size(X,1),1), X];
    a1 = [ones(size((X*Theta1'),1),1), sigmoid(X*Theta1')];
    a2 = sigmoid(a1*Theta2');
    [~,p]=max(a2,[],2);
end