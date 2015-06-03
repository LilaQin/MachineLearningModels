function [best_lambda,test_cost] =RidgeLogisticRegressionCV(trainX,trainY,testX,testY,alpha,epsilon,lambda_set)
% Find the best lambda for Logistic regression based on crossvalidation(CV)
% input: trainX, trainY used to get learning theta 
%        testX,testY are used to choose the best lambda in lambda_set
%        alpha - learning rate
%        epsilon - stopping criterion
%        lambd_set-is a given set of regulation parameter 
% output: best_lambda - the best lambdar in the given set of regulation parameter, when tested in test data 
%         test cost - min least square cost  
%
num_lambda=size(lambda_set,2);% get the number of lambda in the given set
cost_vector=zeros(num_lambda,2);% store the trainning cost and test cost when apply different learning theata and lambda to test data, initialized as zero
                                % cost_vector=[trainning cost, test cost]       
for k=1:+1:num_lambda
 lambda=lambda_set(1,k);  
 [theta,cost_vector(k,1)]=RidgeLogisticRegressionGradientDescent(trainX,trainY,alpha,epsilon,lambda);
 cost_vector(k,2)=CostFunction(testX,testY,theta);
end
figure
plot(lambda_set, cost_vector(:,2)','o',lambda_set, cost_vector(:,1)','*');
title('Training cost And Test cost Plot')
xlabel('lambda')
ylabel('Cost')

[cost,index]= min(abs(cost_vector(:,2)),[],1);
best_lambda=lambda_set(1,index);
test_cost=cost;

end











