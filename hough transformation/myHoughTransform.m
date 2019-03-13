function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)
%Im - grayscale image - 
%threshold - prevents low gradient magnitude points from being included
%rhoRes - resolution of rhos - scalar
%thetaRes - resolution of theta - scalar

%%ignore pixels
[a,b]=size(Im);
Im_threshold=zeros(a,b);
for i0=1:a
    for j0=1:b
        if Im(i0,j0)>=threshold
            Im_threshold(i0,j0)=Im(i0,j0) ;
        end
    end
end

%%hough transform
[M,N]=size(Im_threshold);
D = sqrt(M^2 + N^2);
%q = ceil(D/rhoRes);
%nrho = q;%number of rhoScale
%rhoScale = linspace(-q*rhoRes,q*rhoRes, nrho);
rhoScale =0:rhoRes:D;
nrho=length(rhoScale);
thetaScale=0:thetaRes:2*pi;
ntheta=length(thetaScale);%number of thetaScale
H=zeros(nrho,ntheta);
[Y,X]=find(Im_threshold); 
%slope=(nrho-1)/(rhoScale(end)- rhoScale(1));
for i=1:size(X,1)
    for j=1:ntheta
    rho_vec=X(i)*cos(thetaScale(j))+Y(i)*sin(thetaScale(j));
    if rho_vec>0
    %rho_idx = round(slope*(rho_vec- rhoScale(1)));
        rho_idx=ceil(rho_vec/rhoRes);
        H(rho_idx,j)=H(rho_idx,j)+1;
    end
    end
end     
end
        
        
