classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
    properties
        ClassWeights;
    end
    
    methods
        function layer=weightedClassificationLayer(weights)
            layer.ClassWeights=weights;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 
            % Find observation and sequence dimensions of Y
            S=size(Y);
            
            if ismember(1,size(layer.ClassWeights))
                % Reshape ClassWeights to KxNxS
                W = repmat(permute(layer.ClassWeights(:),[3,2,1]),1,1,1,S(4));
                
                % Compute the loss
                loss = -sum( W(:).*T(:).*log(Y(:)) )/S(4);
            else
                W=layer.ClassWeights;
                WT=W'*squeeze(T);
                loss=sum(sum((squeeze(T-Y).^2.*WT)))/S(4);
            end
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the weighted cross entropy loss with respect to the
            % predictions Y.
            % Find observation and sequence dimensions of Y
            S=size(Y);
            
            if ismember(1,size(layer.ClassWeights))
            % Reshape ClassWeights to KxNxS
            W = repmat(permute(layer.ClassWeights(:),[3,2,1]),1,1,1,S(4));
            
            % Compute the derivative
            dLdY = -(W.*T./Y)/S(4);
            else
                W=layer.ClassWeights;
                WT=W'*squeeze(T);
                dLdY=permute((-2*WT.*squeeze(T-Y))/S(4),[3,4,1,2]);
            end
        end
    end
end

