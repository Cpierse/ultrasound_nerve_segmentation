% Here I will just play around with stuff. From this point, the code will
% die off or be moved to a different script.


%% Find the Artery - the artery is the dark round circle next to the BP.
% There could be two.

% For testing:
img = img1;
img_mask = img1_mask;
imagesc(img);
colormap(gray);

% 1) Apply a low pass Gaussian Filter:
st = 4;
img = lowpassIm(img,st);
imagesc(img)

% 2) Threshold the data based on intensity:
nlevels = 3;
bStack = ttbd(img,nlevels,1);
bStackOne = zeros(size(bStack,1),size(bStack,2),1,class(img));
for i = 1:size(bStack,3); bStackOne = bStackOne+bStack(:,:,i).*i; end
imagesc(bStackOne)
highlightBP(bStackOne,img_mask);


%% Try fractal dimension/box counting:

nlevels = 20;
% First load an image where there is a mask:
img_idx = 1
img_idx = 3

img = imread( strcat(pwd, '\train\', imgs{img_idx}));
img_mask = imread([pwd, '\train\', img_masks{img_idx}]);

st = 15;
img = lowpassIm(img,st);


bStack = ttbd(img,nlevels,1);
bStackOne = zeros(size(bStack,1),size(bStack,2),1,class(img));
for i = 1:size(bStack,3); bStackOne = bStackOne+bStack(:,:,i).*i; end
highlightBP(bStackOne,img_mask);

[rows,cols] = find(img_mask);
b_slice = bStackOne(min(rows):max(rows),min(cols):max(cols))
imagesc(b_slice)


start = round([rand().*(size(bStack,1)-(max(rows)-min(rows))),rand().*(size(bStack,1)-(max(cols)-min(cols)))])
o_slice = bStackOne(1:max(rows)-min(rows),1:max(cols)-min(cols))
o_slice = bStackOne(start(1):start(1)+max(rows)-min(rows),start(2):start(2)+max(cols)-min(cols))
imagesc(o_slice)

f_dim = zeros(1,nlevels);
for i = 1:nlevels
    subplot(1,2,1)
    boxcount(b_slice==i,'slope')
    %boxcount(edge(b_slice==i,'canny'))
    title('Area of interest - slope')
    subplot(1,2,2)
    boxcount(o_slice==i,'slope')
    %boxcount(edge(o_slice==i,'canny'))
    title([num2str(start(1)), ', ' num2str(start(2)), ' Pos Box of same size - slope'])
    pause(2)
    
end
[mean(b_slice(:)), mean(o_slice(:))]
[var(b_slice(:)), var(o_slice(:))]
[length(unique(b_slice(:))), length(unique(o_slice(:)))]


% Observations: The region of interest seems to always have a curve, while
% the random region does not. This means that the number of thresholds
% present or the 


%% Testing Beltrami Stuff:
addpath([pwd '\BeltramiPD'])

s = 20; % noise level

% parameters
beta = 1;
tol = 1e-3;
lambda = 1/s;
r1 = 0.1;
r2 = 0.1/lambda;
Maxit = 2000;

img_dbl = double(img);
img_dbl = 256*(img_dbl-min(img_dbl(:)))/(max(img_dbl(:))-min(img_dbl(:)));
[img_bel,k] = BeltramiPD( img_dbl, beta, lambda, r1, r2, tol, Maxit );

img_lpg = lowpassIm(img,s);

img_wnr = wiener2(img,[s/4 s/4]);

figure('Name', 'Beltrami Denoising');
subplot(2,2,[1]);
imagesc( img ); axis equal; axis off; colormap(gray);
title('Original');

subplot(2,2,2);
imagesc( img_bel ); axis equal; axis off; colormap(gray);
title('Beltrami');

subplot(2,2,4);
imagesc( img_lpg ); axis equal; axis off; colormap(gray);
title('Low-Pass Gauss');

subplot(2,2,3);
imagesc( img_wnr ); axis equal; axis off; colormap(gray);
title('Wiener');


%% Getting the average mask size:


