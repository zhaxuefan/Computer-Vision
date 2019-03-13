function [rhos, thetas] = myHoughLines(H, nLines)
%%first setting
h_new = H;
peaks = [];
done=false;
[m,n]=size(h_new);
for i=2:m-1
    for j=2:n-1
        if max([h_new(i+1,j),h_new(i-1,j),h_new(i,j+1),h_new(i,j-1),h_new(i+1,j+1),h_new(i+1,j-1),h_new(i-1,j+1),h_new(i-1,j-1)])>h_new(i,j)
            h_new(i,j)=0;
        end
    end
end
while ~done
    [max_num,max_idx] = max(h_new(:));
    [p,q] = ind2sub(size(h_new), max_idx);
    peaks=[peaks;[p,q]];
    h_new(p,q)=0;
    if size(peaks,1) == nLines
        done=true;
    else
        done=false;
    end
end
rhos=peaks(:,1);
thetas=peaks(:,2);
end
        