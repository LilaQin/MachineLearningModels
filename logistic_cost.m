function cost=logistic_cost(Y,H_X)
m=size(Y,1);
% m is the number of samples
cost=0;
for k=1:+1:m
   % X_test=sigmoid(X(k,:)*theta);
    if (Y(k,1)==1)
        cost=cost+(-log(H_X(k,1)));
    else
        cost=cost+(-log(1-H_X(k,1)));
    end
end
 cost=cost/(m*1.0 );  
end
