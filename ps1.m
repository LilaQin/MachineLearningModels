%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS184A/284A  PS1
% this is the main program for ps1
% please do not change this file!
% your assignment is to write three functions called by this main program:
% 1: standardize()
% 2: LinearRegressionGradientDescent()
% 3: LinearRegressionNormalEqn()
% the templates for these three functions are provided
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = importdata('parkinsons_updrs.data',',',1);
data = M.data;   % 5875 x 22 matrix
X = data(:,7:end);  % 16 voice measurements
Y = data(:,5);  % symptom score: motor_UPDRS 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem 1:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write a function standardize() to normalize the columns of an input matrix to
% be mean zero and variance 1
%
% use the function to normalize X and Y

X = standardize(X);
Y = standardize(Y);

% use mean(X,1) and std(X,[],1)to confirm that the data is properly
% normalized
mean(X,1)
std(X,[],1)

% same to Y
mean(Y)
std(Y)

% augment X to include the bias term
X = [ones(size(X,1),1),X];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem 2:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write a linear regression function based on gradient descent
%
%

[theta1,cost1]=LinearRegressionGradientDescent(X,Y,0.01,0.00001);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem 3:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write a linear regression function by solving the normal equation
%
%

[theta2,cost2]=LinearRegressionNormalEqn(X,Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem 4:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compare the results obtained from above two methods and discuss
% the differences



