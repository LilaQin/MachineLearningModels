function [clusters,LL] = gaussian_mixture_model(X,K,sigma)
% implement a Gaussian Mixture Model 
% input: X - an m by n matrix
%        K - the number of mixtures (or clusters)
%        sigma - the standard deviation of the Gaussians
%
% return: 
%       clusters - a K by n matrix containing the final means of K Gaussian mixtures
%       LL - the final log likelihood function
%
    [m,n]=size(X);% get the number of data (m) and the dimension of data (n)
    sigmaMatrix=ones(1,n)*sigma;
    %use kmeans initialize GMM model
    
%[initial_assign,meanforN,cost] = kmeans(X,K); % meanforN is the mean for initial Gaussian Distribution
    

    % Initialize GMM model randomly
    P = randperm(m);   % randomly permute 1:m
    cenroids_index = P(1:K);  % randomly choose k Points as initial centroids of K clusters
    meanforN=X(cenroids_index,:);%initialize centroids, assign cluster centroids with randoml chosen points 
    oldsumloglikelihood=0;
    sumloglikelihood=0;
    threshold = 1e-4;   %Randomly chosen convergence threshold
    iterationNum=50;% records iteration number
    % differVec=zeros(1,iterationNum);
    sumloglikelihoodVec=[];
    
    iteration=1;
   while( iteration<=iterationNum )
       
    % E-step calculate posterior probabilaty

        for k=1:K
         Xi_NormalPro(i,k)=mvnpdf(X(i,:),meanforN(k,:),sigmaMatrix);% the probability of ith data assgined to a Gaussian distribution

        end
            likelihood = (sum(Xi_NormalPro,2)*1.0/K);
            EXP=Xi_NormalPro./(ones(K,1)*likelihood')';
            EXP=EXP/K;
    end
    % M_stemp update P_Z_K and mean of each Guassian Distribution
    %updat P_Z_K

         %   P_Z_K=sum(P_Zi_Xi,k)'*1.0/m;
    
   %update the mean of each Guassian Distribution 
   for h=1:K
       sum1=0;
       sum2=0;
        for i=1:m
            sum1=sum1+EXP(i,h)*X(i);
            sum2=sum2+EXP(i,h);
%             sum1=sum1+P_Zi_Xi(i,h)*X(i);
%             sum2=sum2+P_Zi_Xi(i,h);
        end
        meanforN(h)=sum1*1.0/sum2;
   end
   % calculate log likelihood 
   sumloglikelihood=mean(log(likelihood));
   
   
   diff=abs(sumloglikelihood- oldsumloglikelihood);
   if(iteration==1)
   sumloglikelihoodVec(1,iteration)=sumloglikelihood;
   else
       sumloglikelihoodVec=[sumloglikelihoodVec,sumloglikelihood];
   end
   if  (diff<threshold)
      clusters=meanforN;
      LL=sumloglikelihood;
      break;
    
   else
       oldsumloglikelihood=sumloglikelihood;
   end
   iteration=iteration+1;
   
   end
   LL=sumloglikelihood;
   figure(1)
   plot(1:size(sumloglikelihoodVec,2),sumloglikelihoodVec(1,:),'*');
end
