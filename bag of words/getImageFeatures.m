function [h] = getImageFeatures(wordMap, dictionarySize)
% Convert an wordMap to its feature vector. In this case, it is a histogram
% of the visual words
% Input:
%   wordMap:            an H * W matrix with integer values between 1 and K
%   dictionarySize:     the total number of words in the dictionary, K
% Outputs:
%   h:                  the feature vector for this image
%     h=zeros(1,dictionarySize);
%     [H,W]=size(wordMap);
%     for n=1:dictionarySize
%         x=nnz(wordMap==n);
%         h(n)=x/(H*W);
%     end
% %     histogram(h,'Normalization','probability');
    %h=zeros(1,dictionarySize);
    [H,W]=size(wordMap);
    newwordMap=reshape(wordMap,1,H*W);
    h=hist(newwordMap,dictionarySize);
    h=h/(H*W);
    
    

	% -----fill in your implementation here --------

    

    % ------------------------------------------

end
