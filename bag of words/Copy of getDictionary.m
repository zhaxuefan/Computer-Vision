function [dictionary] = getDictionary(imgPaths, alpha, K, method)
% Generate the filter bank and the dictionary of visual words
% Inputs:
%   imgPaths:        array of strings that repesent paths to the images
%   alpha:          num of points
%   K:              K means parameters
%   method:         string 'random' or 'harris'
% Outputs:
%   dictionary:         a length(imgPaths) * K matrix where each column
%                       represents a single visual word
    % -----fill in your implementation here --------
    load('traintest.mat');
    namenum=numel(all_imagenames);
    imagepoints=[];
    filterBank = createFilterBank();
    pixelResponses=[];
    for i1=1:namenum
        name=all_imagenames{1,i1};
        img=imread(sprintf('%s/%s', imgPaths,name));
        filterResponses = extractFilterResponses(img, filterBank);
        [~,~,n]=size(filterResponses);
        for i2=1:n
            if isequal(method,'random')
                points = getRandomPoints(filterResponses(:,:,i2), alpha);
            else
                points = getHarrisPoints(filterResponses(:,:,i2), alpha, 0.04);
            end
            for i3=1:alpha
                imagepoints(i3,1)=filterResponses(points(i3,1),points(i3,2),i2);
            end
            pixelResponses(alpha*i1-alpha+1:alpha*i1,i2)=imagepoints;
        end  
    end
    [~, dictionary] = kmeans(pixelResponses, K, 'EmptyAction', 'drop');
    % ------------------------------------------
    
end
