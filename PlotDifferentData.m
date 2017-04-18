% Here we explore and plot the data in different ways!
N_preview = 100


%% Plot a few images and their masked versions:
figure(1)
for img_idx = 1:N_preview
subplot(1,2,1)
img = imread( strcat(pwd, '\train\', imgs{img_idx}));
imshow(img)
colormap(gray)
subplot(1,2,2)
img_mask = imread([pwd, '\train\', img_masks{img_idx}]);
img1_masked = img.*(img_mask./255);
imshow(img1_masked)
colormap(gray)
pause(0.5)
end
close all
% check out 10_104.tif

%% Plot images with edge detection on mask:
figure(1)
for img_idx = 1:N_preview
img = imread( strcat(pwd, '\train\', imgs{img_idx}));
img_mask = imread([pwd, '\train\', img_masks{img_idx}]);
highlightBP(img,img_mask);

pause(0.2)
if any(any(img_mask))
   pause(0.5) 
end
hold off
end


%% Plot multiple images thresholded:
figure(1)
nlevels = 4; nlevels = 16
low_pass = 0; bel = 1;
if bel, addpath([pwd '\BeltramiPD']), end
threshold = 1; threshold = 0

for img_idx = 1:N_preview
img = imread( strcat(pwd, '\train\', imgs{img_idx}));
img_mask = imread([pwd, '\train\', img_masks{img_idx}]);

if low_pass
    st = 4; %st = 8;
    img = lowpassIm(img,st);
end
if bel
    beta = 1;
    tol = 1e-3;
    lambda = 1/10;
    r1 = 0.1;
    r2 = 0.1/lambda;
    Maxit = 2000;
    img_dbl = double(img);
    img_dbl = 256*(img_dbl-min(img_dbl(:)))/(max(img_dbl(:))-min(img_dbl(:)));
    [img,k] = BeltramiPD( img_dbl, beta, lambda, r1, r2, tol, Maxit );
end

if threshold
bStack = ttbd(img,nlevels,1);
bStackOne = zeros(size(bStack,1),size(bStack,2),1,class(img));
for i = 1:size(bStack,3); bStackOne = bStackOne+bStack(:,:,i).*i; end
highlightBP(bStackOne,img_mask);
else
highlightBP(img,img_mask);
end


pause(0.2)
if any(any(img_mask))
   pause(0.8) 
end
hold off
end



