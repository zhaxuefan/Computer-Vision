function [points] = getRandomPoints(I, alpha)
% Generates random points in the image
% Input:
%   I:                      grayscale image
%   alpha:                  random points
% Output:
%   points:                    point locations
%
	% -----fill in your implementation here --------
    [i,j]=size(I);
    points=zeros(alpha,2);
    xp=randi(i,1,alpha);
    yp=randi(j,1,alpha);
    imagepoints={};
    points(:,1)=xp;
    points(:,2)=yp;
%     imshow(I);
%     hold on
%     plot(points(:,2),points(:,1),'r*');
    % ------------------------------------------

end

