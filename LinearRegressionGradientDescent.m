function [theta,cost] = LinearRegressionGradientDescent(X,Y,alpha,epsilon)
% function [theta,cost] = LinearRegressionGradientDescent(X,Y,alpha,epsilon)
% linea regression based on gradient descent
% input: X - feature matrix, each row represents one sample, first
%            column is always 1 
%        Y - output vector, each row represents one sample 
%        alpha - learning rate
%        epsilon - stopping criterion
% 
% output: theta - regression coeffients 
%         cost - min least square cost 
%



% randomly initialize theta
theta = randn(size(X,2),1);
m = size(X,1); % the number of samples

n = 1;
while 1,
  grad = X'*(X*theta - Y)/m;
  theta = theta - alpha * grad;
  residule = Y - X*theta;
  cost = residule'*residule/(2*m);
  if mod(n,100)==0,
    fprintf(1,'n-iter: %d\tcost:%f\n',n,cost);
  end
  
  if norm(grad)<epsilon, break; end
  n = n + 1;
end














