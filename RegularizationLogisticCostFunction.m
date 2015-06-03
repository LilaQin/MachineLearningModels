function cost=RegularizationLogisticCostFunction(X,Y,theta,lambda)
m=size(X,1);
% m is the number of samples
cost=0;
for k=1:+1:m
    X_test=sigmoid(X(k,:)*theta);
    if (Y(k,1)==1)
        cost=cost+(-log(X_test));
    else
        cost=cost+(-log(1-X_test));
    end
end
 cost=cost/(m*1.0 )+lambda/2 * sum(theta'*theta);  
end
