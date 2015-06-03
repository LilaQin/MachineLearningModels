function [output,cost] = mlp(trainX,trainY,testX,testY,alpha,minibatchsize,nepochs)
%this is a three-layer neural network
%the number of units in the hidden layer be 5

layerCount=3; %the total of layers in this nueral network
unitCount=5;% the number of units in the hidden layer
[m,inputArgumentCount]=size(trainX); % [the number of samples,the number of input arguments except offset]
layerOfUnits=[inputArgumentCount+1,unitCount+1,1];% layerOfUnits=[9+1,5+1,1]because we include the weight of offset
%---Set initial random weights
    thetaCell = cell(1, layerCount-1);
    for i = 1:layerCount-1
            thetaCell{i} = unifrnd(-1, 1, layerOfUnits(i),layerOfUnits(i+1));
       % end
    end

   % fpInput=unitOutputCell{1};
   cost_vector=zeros(nepochs,2);% used to stroe cost: cost_vector=[train_cost,test_cost]
   P = randperm(m); 
   for k=1:+1:nepochs

      
      for iter=1:+1:floor(m/minibatchsize)
          index=P(((iter-1)*minibatchsize+1):iter*minibatchsize);
          X=trainX(index,:);
          Y=trainY(index,:);
    %forwardpropagation
    aOutputCell=ForwardPropagation(X, thetaCell, layerCount,minibatchsize);  
    
    
    % backpropagation
    thetaCell=BackwardPropagation(aOutputCell,Y,thetaCell, layerCount,minibatchsize,alpha);
   % [prev_delta_Cell,thetaCell]=BackwardPropagation(aOutputCell,Y,thetaCell, layerCount,minibatchsize,alpha,prev_delta_Cell);
      end
    %after getting new theta, calculate cost regard to trainData
    aOutputCell=ForwardPropagation(X, thetaCell, layerCount,size(X,1));
    cost_vector(k,1)=logistic_cost(Y, aOutputCell{layerCount}); 
    %use test data to test the model
   aOutputCell=ForwardPropagation(testX, thetaCell, layerCount,size(testX,1));
   %calculate cost when use trainData
   cost_vector(k,2)=logistic_cost(testY, aOutputCell{layerCount}); 
    
   end
    cost=cost_vector(nepochs,2); 
   %compare cost after each epoch
   figure
   plot(1:nepochs, cost_vector(:,1)','o',1:nepochs, cost_vector(:,2)','*');
   title('Training cost And Test cost Plot')
    xlabel('nepochs')
    ylabel('Cost')
 
    
    
    
    %caluate the final bianry classification output
    output=aOutputCell{layerCount};
    idx= output>=0.5;
    output(idx)=1;
    idx= output<0.5;
    output(idx)=0;
    % calculate how many wrong predictions, compare to test Y
    d=output-testY;
    d=d.*d;
    wrong=sum(d);

end
