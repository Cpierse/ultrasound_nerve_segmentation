function [bStack] = ttbd(img,nlevels,alt)
% Ttbd creates a stack of nlevel binary thresholded images from a single 
% image img. The alt method does not require a newer matlab:
% Credit: Conrad Foo
if nargin ==2
    alt = 0
end
if ~alt
    thresh = multithresh(img,nlevels-1);
    bStack = zeros(size(img,1),size(img,2),nlevels,'like',img);
    bStack(:,:,1) = bsxfun(@lt,img,thresh(1));
    for i = 1:nlevels-2
        bStack(:,:,i+1) = bsxfun(@gt,img,thresh(i)).*bsxfun(@lt,img,thresh(i+1));
    end
    bStack(:,:,nlevels) = bsxfun(@gt,img,thresh(end));
elseif alt
    thresh = otsu(img,nlevels);
    bStack = zeros(size(img,1),size(img,2),nlevels,class(img));
    for i = 1:nlevels
         bStack(:,:,i) = (thresh==i);
    end
    
end
end
