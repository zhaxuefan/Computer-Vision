function [filterResponses] = extractFilterResponses(I, filterBank)
% CV Fall 2018 - Provided Code
% Extract the filter responses given the image and filter bank
% Pleae make sure the output format is unchanged.
% Inputs:
%   I:                  a 3-channel RGB image with width W and height H
%   filterBank:         a cell array of N filters
% Outputs:
%   filterResponses:    a HxWx3N matrix of filter responses


    %Convert input Image to Lab
    doubleI = double(I);
    if length(size(doubleI)) == 2
        tmp = doubleI;
        doubleI(:,:,1) = tmp;
        doubleI(:,:,2) = tmp;
        doubleI(:,:,3) = tmp;
    end
    [L,a,b] = RGB2Lab(doubleI(:,:,1), doubleI(:,:,2), doubleI(:,:,3));
    h = size(I,1);
    w = size(I,2);

   
    % -----fill in your implementation here --------
    nfilter=size(filterBank);%cell array length
    M=zeros(h,w,3*nfilter(1));
    for n=1:nfilter(1)
        Lnew=imfilter(L,filterBank{n,1});
        anew=imfilter(a,filterBank{n,1});
        bnew=imfilter(b,filterBank{n,1});
        M(:,:,(3*n-2))=Lnew;
        M(:,:,(3*n-1))=anew;
        M(:,:,3*n)=bnew;
    end
    filterResponses=M;
    % ------------------------------------------
end
