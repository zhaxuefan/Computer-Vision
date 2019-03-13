function [points] = getHarrisPoints(I, alpha, k)
% Finds the corner points in an image using the Harris Corner detection algorithm
% Input:
%   I:                      grayscale image
%   alpha:                  number of points
%   k:                      Harris parameter
% Output:
%   points:                    point locations
%
    % -----fill in your implementation here --------
%         Inew=I;
%     if (ndims(I) == 3)
%         I = rgb2gray(I);
%     end
%     I = double(I) / 255;
    rows = size(I, 1);
    columns = size(I, 2);
    %sigma = 1;
%%Compute x and y derivatives of image
    Gy = fspecial('sobel');
    Gx=Gy';
    Ix = imfilter(I, Gx,'replicate','conv');
    Iy = imfilter(I, Gy,'replicate','conv');
    %Gxy = fspecial('gaussian',max(1,fix(6*sigma)), sigma); % Gaussien Filter
    Gxy = ones(3,3);
%% Compute products of derivatives at every pixel
    Ix2 = Ix .^ 2;
    Iy2 = Iy .^ 2;
    Ixy = Ix .* Iy;
%% Compute the sums of the products of derivatives at each pixel
    Sx2 = imfilter(Ix2,Gxy,'conv');
    Sy2 = imfilter(Iy2,Gxy,'conv');
    Sxy = imfilter(Ixy,Gxy,'conv');
%% Define at each pixel(x, y) the matrix H
    im=zeros(rows,columns);
    for x=1:rows
        for y=1:columns
            H = [Sx2(x,y) Sxy(x,y); Sxy(x,y) Sy2(x,y)];
            R = det(H) - k * (trace(H)^2);
            im(x, y) = R;
        end
    end
    points=[];
    for m1=1:alpha
        [~,max_idx]=max(im(:));
        [p,q]=ind2sub(size(im),max_idx);
        points=[points;[p,q]];
        im(p,q)=0;  
    end
%      imshow(Inew);
%      hold on
%      plot(points(:,2),points(:,1),'r*');
    % ------------------------------------------
    
end
