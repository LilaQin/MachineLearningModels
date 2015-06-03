function M = standardize(M)
% function M = standardize(M)
% standardize each column of M to be mean zero and variance 1
%M=zscore(M);
 M_mean=mean(M);
 M_std=std(M);
 [m,n]=size(M);
 for k=1:+1:n;
     M(:,k)=M(:,k)-M_mean(k);
     M(:,k)=M(:,k)/M_std(k);
 end
end







