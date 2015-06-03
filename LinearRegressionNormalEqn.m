function [theta,cost] = LinearRegressionNormalEqn(X,Y)
% function [theta,cost] = LinearRegressionNormalEqn(X,Y)
% linea regression based on gradient descent
% input: X - feature matrix, each row represents one sample, first
%            column is always 1 
%        Y - output vector, each row represents one sample 
% 
% output: theta - regression coeffients 
%         cost - min least square cost 
%


theta = pinv(X'*X)*X'*Y;
residual = Y - X*theta;
cost = residual'*residual/(2*size(X,1));
















