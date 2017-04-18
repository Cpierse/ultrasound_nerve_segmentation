function [img] = highlightBP(img,img_mask)
%Given a B&W image and the image mask, this program should highlight the BP 
%Nerves. This can be used on the raw images or thresholded images.

% First find the edges:
B = edge(img_mask,'canny'); % imagesc(B)
% Simply infuse them and use the difference to highlight the image:
if any(B(:)), img = imfuse(img,B,'diff'); end
imagesc(img)
colormap(gray);


end

% % Create a color image.
% img_color = zeros(size(img,1),size(img,2),3,class(img));
% % Fill all channels
% for j = 1:3; img_color(:,:,j) = img; end
% % In this process, the scaling now needs to match the class:
% if strmatch(class(img),'double'); 
% else
%     img_color = img_color.*(intmax(class(img_color))/max(img(:)+1));
% end
% % Where B is present, max out channel.
% if any(any(B))
%     img_color(:,:,1) = img_color(:,:,1) + (cast(B,class(img)).*max(img_color(:)));
% end
% 
% if strmatch(class(img),'double')
%     img_color = img_color./max(max(max(img_color)));
% end
% 
% imagesc(img_color) 