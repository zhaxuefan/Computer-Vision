function [wordMap] = getVisualWords(I, dictionary, filterBank)
% Convert an RGB or grayscale image to a visual words representation, with each
% pixel converted to a single integer label.   
% Inputs:
%   I:              RGB or grayscale image of size H * W * C
%   filterBank:     cell array of matrix filters used to make the visual words.
%                   generated from getFilterBankAndDictionary.m
%   dictionary:     matrix of size 3*length(filterBank) * K representing the
%                   visual words computed by getFilterBankAndDictionary.m
% Outputs:
%   wordMap:        a matrix of size H * W with integer entries between
%                   1 and K

    % -----fill in your implementation here --------
    filterResponses = extractFilterResponses(I, filterBank);
    [K,~]=size(dictionary);
    [H,W,N]=size(filterResponses);
    wordMap=zeros(H,W);
    new=reshape(filterResponses,H*W,60);
    D=pdist2(new,dictionary,'euclidean');
    [~,ind]=min(D');
    wordMap=reshape(ind,H,W);
    % ------------------------------------------
end
