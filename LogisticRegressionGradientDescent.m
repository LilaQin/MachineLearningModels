function [theta,cost] = LogisticRegressionGradientDescent(X,Y,alpha,epsilon)
% function [theta,cost] = LogisticRegressionGradientDescent(X,Y,alpha,epsilon)
% based on gradient descent
% input: X - feature matrix, each row represents one sample, first
%            column is always 1 
%        Y - output vector, each row represents one sample 
%        alpha - learning rate
%        epsilon - stopping criterion
% 
% output: theta - regression coeffients 
%         cost - min least square cost 
%
m=size(Y,1);% m is the number of samples
n=size(X,2);% n is the number of features
theta=zeros(n,1); % initialize theta as 0
temp_theta=theta;
 
while(1)
H=sigmoid(X*theta); % get the model of logstic regression
for j=1:+1:n
Colum_X=(X(:,j))';
%temp_theta(j,1)=theta(j,1)-alpha*sum(Colum_X*((T-Y).^2))/m;
temp_theta(j,1)=theta(j,1)-alpha*sum(Colum_X*(H-Y))/m;
end
%max_diff=max(abs(temp_theta-theta));
grad=(theta-temp_theta)*1.0/alpha;% this is gradient
theta=temp_theta;
%if(max_diff<epsilon)%
if(norm(grad)<epsilon)
    break;
end
end
H=sigmoid(X*theta); 
delt2=H-Y;
cost=CostFunction(X,Y,theta);
end
