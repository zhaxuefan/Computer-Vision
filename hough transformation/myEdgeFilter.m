function [Im Io Ix Iy] = myEdgeFilter(img, sigma)
%%smooth image
image=img;
hsize=2*ceil(3*sigma)+1;
kernel = fspecial('gaussian',hsize,sigma);
image_smooth=myImageFilter(image,kernel);

%%find gradient
Gy_sobel = fspecial('sobel');
Gx_sobel=Gy_sobel';
Ix=myImageFilter(image_smooth,Gx_sobel);
Iy=myImageFilter(image_smooth,Gy_sobel);
Im=sqrt(Ix.^2+Iy.^2);
Io=180*atan2(Ix,Iy)/pi;

%%non maximal surpress
[m,n]=size(Im);
Iom=Io;
for i=2:m-1
    for j=2:n-1
        angle=Io(i,j)+180;
          if (angle>=340 || angle<=22.5) || (angle>=157.5 && angle<=202.5)%90
              Iom(i,j)=90;
          elseif (angle>22.5 && angle<=67.5) || (angle>202.5 && angle<=247.5)%135
              Iom(i,j)=135;
          elseif (angle>67.5 && angle<=112.5) || (angle>247.5 && angle<=292.5)%0
              Iom(i,j)=0;
          else 
              Iom(i,j)=45;
          end
          
        if Im(i,j)~=0
            if Iom(i,j)==0%0
                if (Im(i,j+1)>Im(i,j))||(Im(i,j-1)>Im(i,j))
                    Im(i,j)=0;
                end
                m1=i;n1=j;
                while n1>0 && Iom(m1,n1)==0
                    if Im(i,j)~=max(Im(i,j),Im(m1,n1))
                        Im(i,j)=0;
                    end
                    n1=n1-1;
                end
            elseif Iom(i,j)==45%45
                if (Im(i-1,j+1)>Im(i,j))||(Im(i+1,j-1)>Im(i,j))
                    Im(i,j)=0;
                end
                   m1=i;n1=j;
                while n1>0 && Iom(m1,n1)==45
                    if Im(i,j)<Im(m1,n1)
                        Im(i,j)=0;
                    end
                    m1=m1-1;
                    n1=n1+1;
                end
            elseif Iom(i,j)==90%90
                if (Im(i+1,j)>Im(i,j))||(Im(i-1,j)>Im(i,j))
                    Im(i,j)=0;
                end
                 m1=i;n1=j;
                while n1>0 && Iom(m1,n1)==90
                    if Im(i,j)<Im(m1,n1)
                        Im(i,j)=0;
                    end
                    m1=m1-1;
                end
            else%135
                if (Im(i+1,j+1)>Im(i,j))||(Im(i-1,j-1)>Im(i,j))
                    Im(i,j)=0;
                end
                 m1=i;n1=j;
                while n1>0 && Iom(m1,n1)==135
                    if Im(i,j)<Im(m1,n1)
                        Im(i,j)=0;
                    end
                    m1=m1-1;
                    n1=n1-1;
                end
            end
        end
    end
end
Im(:,[1,n])=0;
Im([1,m],:)=0;
%Im=min(Im(:))+(Im-min(Im(:)))/(max(Im(:))-min(Im(:)));
end
    
                
        
        
