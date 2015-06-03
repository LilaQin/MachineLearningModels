function [theta,cost] =RidgeLogisticRegressionGradientDescent(newX,Y,alpha,epsilon,lambda)
% Ridge Logistic regression based on gradient descent
% input: newX - feature matrix, each row represents one sample, first
%            column is always 1 
%        Y - output vector, each row represents one sample 
%        alpha - learning rate
%        epsilon - stopping criterion
%        lambda: regularization parameter
% output: theta - regression coeffients 
%         cost - min least square cost 
%
m=size(Y,1);% m is the number of samples
n=size(newX,2);% n is the number of features
theta=zeros(n,1); % initialize theta as 0
temp_theta=theta;
 
while(1)
H=sigmoid(newX*theta); % get the model of logistic regression
for j=1:+1:n
Colum_X=(newX(:,j))';
%temp_theta(j,1)=theta(j,1)-alpha*sum(Colum_X*((T-Y).^2))/m;
temp_theta(j,1)=theta(j,1)-alpha*(sum(Colum_X*(H-Y))*1.0/m+lambda*theta(j,1));
end
grad=(theta-temp_theta)*1.0/alpha;% this is gradient
theta=temp_theta;
th=norm(grad);
if(th<epsilon)%
    break;
end
end
H=sigmoid(newX*theta); 
delt2=H-Y;
cost=RegularizationLogisticCostFunction(newX,Y,theta,lambda);
end











