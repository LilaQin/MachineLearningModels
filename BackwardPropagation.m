    function thetaCell=BackwardPropagation(unitOutputCell,trainY,thetaCell, layerCount,sampleNo,alpha)
   % function [prev_delta_Cell,thetaCell]=BackwardPropagation(unitOutputCell,trainY,thetaCell, layerCount,sampleNo,alpha,prev_delta_Cell)
    deltaCell=cell(1,layerCount);
   
    for i=layerCount-1:-1:1
        preOutput=unitOutputCell{i};
        if (i==layerCount-1)%Back propagate for Output layers
            % delta=(d(J)/d(z))*g(z)*(1-g(z))
         deltaCell{layerCount}=(unitOutputCell{layerCount}-trainY).*unitOutputCell{layerCount}.*(1-unitOutputCell{layerCount});

        deltaTheta=(preOutput'*deltaCell{layerCount})*1.0/sampleNo;
        else %Back propagate for Hidden layers
 
        deltaCell{i+1}= (deltaCell{i+2})*(thetaCell{i+1})'.*unitOutputCell{i+1}.*(1-unitOutputCell{i+1}); 

        deltaTheta=(preOutput'*deltaCell{i+1})*1.0/sampleNo;
        end
        
        %update theta
        %momentum
        %prev_delta_Cell{i}=0.1*prev_delta_Cell{i}+deltaTheta;
               
       % thetaCell{i}=thetaCell{i}-alpha*prev_delta_Cell{i};
        %momentum
       
       
        thetaCell{i}=thetaCell{i}-alpha*deltaTheta;
    end
    end
