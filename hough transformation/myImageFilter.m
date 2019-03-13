function [img1] = myImageFilter(img0, h)
%%Input image data and save image and kernel size,reminder if kernel size is not odd
 Image=img0;
 [image_length,image_height]=size(Image);
 [kernel_length,kernel_height]=size(h);
 if (kernel_length~=kernel_height)||(rem(kernel_length+1,2)~=0)
    error('kernel size is wrong,kernel size must be odd')
 end
 kernel_size=kernel_length;
 
 %%padding
 padding_number=(kernel_size-1)/2;
 Image_padded=zeros(image_length+padding_number*2,image_height+padding_number*2);
 Image_padded(padding_number+1:padding_number+image_length,padding_number+1:padding_number+image_height)=Image;
 %Image_padded=padarray(Image,[padding_number padding_number],'replicate');
 [new_length,new_height]=size(Image_padded);
 
 %%convolution
 %flip kernel
 kernel=fliplr(flipud(h));
 for i=1:new_length-kernel_size+1
     for j=1:new_height-kernel_size+1
         img1(i,j)=sum(sum(kernel.*Image_padded(i:i+kernel_size-1,j:j+kernel_size-1)));
     end
 end
end
