    function unitOutputCell=ForwardPropagation(fpInput, thetaCell, layerCount, sampleNo)
    unitOutputCell=cell(1,layerCount); %store the output of every layer
    unitOutputCell{1}=[ones(sampleNo,1),fpInput];% in 1st layer, the output of units is the sample features

    for i=1:layerCount-1
    z=unitOutputCell{i}*thetaCell{i};
    unitOutputCell{i+1}=sigmoid(z);
    end
    end
