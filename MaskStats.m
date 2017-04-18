%Mask Stats-
%Here we analyze the masks in order to understand the data and our approach
%better. Some key questions include:
%1) What proportion of the training data has a valid mask?
%2) What is the averagse size of the mask?
%Assumes the img locations are stored as imgs{i} and img_masks{i};

%% First load all masks into memory and convert to binary.
% Load one for the dimensions:
img_mask = imread( strcat(pwd, '\train\', img_masks{1}));
% Initialize the Mask holder:
IMG_MASKS = false(size(img_mask,1),size(img_mask,2),length(img_masks));
mask_idxs = []; % Find index of those with non-zero masks
for idx = 1:size(IMG_MASKS,3)
    img_mask = logical(imread( strcat(pwd, '\train\', img_masks{idx})));
    IMG_MASKS(:,:,idx) = img_mask;
    if any(img_mask(:));
        mask_idxs(end+1) = idx;
    end
end
display([num2str(round(length(mask_idxs)./size(IMG_MASKS,3)*100)) '% of training data have a mask']);

%% From those that have a mask, find the average height, width, and area (location?)
X_len = zeros(1,length(mask_idxs)); xmax = size(img_mask,1);
Y_len = zeros(1,length(mask_idxs)); ymax = size(img_mask,2);
Areas = zeros(1,length(mask_idxs));
X_cm = zeros(1,length(mask_idxs));
Y_cm = zeros(1,length(mask_idxs));
Theta = zeros(1,length(mask_idxs));
m_idx = 0;
for idx = mask_idxs
    img_mask = IMG_MASKS(:,:,idx);
    m_idx = m_idx+1;
    [rows,cols] = find(img_mask);
    x0 = min(rows); x1 = max(rows); X_len(m_idx) = x1-x0;
    y0 = min(cols); y1 = max(cols); Y_len(m_idx) = y1-y0;
    Areas(m_idx) = sum(img_mask(:));
    X_cm(m_idx) = sum(sum(meshgrid(1:xmax,zeros(1,ymax))'.*img_mask))./sum(img_mask(:));
    Y_cm(m_idx) = sum(sum(meshgrid(1:ymax,zeros(1,xmax)).*img_mask))./sum(img_mask(:));
    [Xs,Ys] = find(img_mask); [THETA] = polyfit(Xs,Ys,1); 
    Theta(m_idx)= atan(THETA(1))*180/pi;
end
% HOLY SHIT THIS Theta LIST MIGHT BE WRONG BECAUSE X AND Y ARE SWITCHED.
% It's not really worth fixing. Seems like it's from the norm now. But
% since I iterate on the final Theta below due to a switch in frame, there
% is no need for concern.
figure(1)
subplot(2,3,1)
hist(X_len); title('X lengths')
subplot(2,3,2)
hist(Y_len); title('Y lengths')
subplot(2,3,3)
hist(Areas); title('Areas')
subplot(2,3,4)
hist(X_cm); title('X centers')
subplot(2,3,5)
hist(Y_cm); title('Y centers')
subplot(2,3,6)
hist(Theta,20); %hist(Theta( and((Theta>-40),(Theta<60))),20); 
title('Thetas')

% Set Theta to zero for now since the data is not rotated.
THETA = 0;
% Make the width and height large enough to cover 80% of the masks X and Y
cutoff = 0.8;
XLEN = round(fminsearch( @(cut) abs(sum(X_len<cut)./sum(X_len>0)-cutoff),100)); 
YLEN = round(fminsearch( @(cut) abs(sum(Y_len<cut)./sum(Y_len>0)-cutoff),100));
save('XY_cm.mat','X_cm','Y_cm','XLEN','YLEN','THETA','Theta','mask_idxs')
%loads X_cm, Y_cm, XLEN = 140; YLEN = 96; THETA = 22; mask_idxs

% Try a mean theta:
figure(2)
THETA = mean(Theta); XCM = mean(X_cm), YCM = mean(Y_cm);
i = 0; 
for idx = mask_idxs(ceil(rand(1,12).*length(mask_idxs)))
    figure(2)
   i = i+1; subplot(4,3,i);
   img_mask = IMG_MASKS(:,:,idx); 
   img_mask = imrotate(img_mask,THETA);
   %img_mask = rotateAround(img_mask,YCM,XCM,theta);
   imagesc(img_mask)
end

%imagesc(imrotate(img.*uint8(img_mask),theta))


%% Stats for rotated images:
X_len = zeros(1,length(mask_idxs)); xmax = size(img_mask,1);
Y_len = zeros(1,length(mask_idxs)); ymax = size(img_mask,2);
Areas = zeros(1,length(mask_idxs));
X_cm = zeros(1,length(mask_idxs));
Y_cm = zeros(1,length(mask_idxs));
m_idx = 0;
THETA = 14.4; % Calculated above
THETA = 22; % Rough fix because of change of axis.
ThetaR = zeros(1,length(mask_idxs));
for idx = mask_idxs
    img_mask = IMG_MASKS(:,:,idx);
    %img_mask = rotateAround(img_mask,mean(Y_cm),mean(X_cm),theta);
    img_mask = imrotate(img_mask,THETA);
    m_idx = m_idx+1;
    [rows,cols] = find(img_mask);
    x0 = min(rows); x1 = max(rows); X_len(m_idx) = x1-x0;
    y0 = min(cols); y1 = max(cols); Y_len(m_idx) = y1-y0;
    Areas(m_idx) = sum(img_mask(:));
    X_cm(m_idx) = sum(sum(meshgrid(1:xmax,zeros(1,ymax))'.*img_mask))./sum(img_mask(:));
    Y_cm(m_idx) = sum(sum(meshgrid(1:ymax,zeros(1,xmax)).*img_mask))./sum(img_mask(:));
    [Xs,Ys] = find(img_mask); [thetar] = polyfit(Xs,Ys,1); 
    ThetaR(m_idx)= atan(thetar(1))*180/pi;
end
figure(1)
subplot(2,3,1)
hist(X_len); title('X lengths')
subplot(2,3,2)
hist(Y_len); title('Y lengths')
subplot(2,3,3)
hist(Areas); title('Areas')
subplot(2,3,4)
hist(X_cm); title('X centers')
subplot(2,3,5)
hist(Y_cm); title('Y centers')
subplot(2,3,6)
hist(ThetaR( and((ThetaR>-40),(ThetaR<60))),20); title('Thetas')

% Results:
XCM_R = round(mean(X_cm)); %229
YXM_R = round(mean(Y_cm)); %366
% Make the width and height large enough to cover 80% of the masks X and Y
cutoff = 0.8;
XLEN = round(fminsearch( @(cut) abs(sum(X_len<cut)./sum(X_len>0)-cutoff),100)); %140
YLEN = round(fminsearch( @(cut) abs(sum(Y_len<cut)./sum(Y_len>0)-cutoff),100)) %96

%% Test for processing data the way needed for NN:
% First make sure rescaling is good.
i = 0;
for idx = mask_idxs(1:12)
    i = i + 1;
    img_mask = IMG_MASKS(:,:,idx);
    img = imread( strcat(pwd, '\train\', imgs{idx}));
    img_w_mask = imrotate(img.*uint8(img_mask),THETA);
    img_w_mask = imresize(img_w_mask,0.2);
    subplot(3,4,i)
    imagesc(img_w_mask)
    colormap(gray)
end

% Cool, not bad. Apply Preprocessing Filters.
i = 0;
addpath([pwd '\BeltramiPD']); 
beta = 1; tol = 1e-3; lambda = 1/10;
r1 = 0.1; r2 = 0.1/lambda; Maxit = 2000;
for idx = mask_idxs(1:12)
    i = i + 1;
    img_mask = IMG_MASKS(:,:,idx);
    img = imread( strcat(pwd, '\train\', imgs{idx}));
    img_w_mask = imrotate(img.*uint8(img_mask),THETA);
    img_w_mask = double(img_w_mask);
    img_w_mask = 256*(img_w_mask-min(img_w_mask(:)))/(max(img_w_mask(:))-min(img_w_mask(:)));
    [img,k] = BeltramiPD( img_w_mask, beta, lambda, r1, r2, tol, Maxit );
    img_w_mask = imresize(img_w_mask,0.2);
    subplot(3,4,i)
    imagesc(img_w_mask)
    colormap(gray)
end

% Alright! We can always change the preprocessing filter later.

%% Clear memory of the huge mask array:
clear IMG_MASKS

%% Create generic mask and apply at X_cm Y_cm:
% Global variables previuosly extracted:
% XLEN = 140; YLEN = 96; THETA = 22; 
SCALE = 0.2;
N_samples = length(imgs);
% Load each image. 
% If masked, create new image with mask at its X_cm, Y_cm;
% If not masked, create new imahe with mask at ranom X_cm, Y_cm;
for img_idx = 1:N_preview
    % Load the data:
    img = imread( strcat(pwd, '\train\', imgs{img_idx}));
    % Rotate the data:
    img = imrotate(img,THETA)

    
    img_w_mask = imrotate(img.*uint8(img_mask),THETA);
    img_w_mask = double(img_w_mask);
    img_w_mask = 256*(img_w_mask-min(img_w_mask(:)))/(max(img_w_mask(:))-min(img_w_mask(:)));
    [img,k] = BeltramiPD( img_w_mask, beta, lambda, r1, r2, tol, Maxit );
    img_w_mask = imresize(img_w_mask,0.2);
    
    
    
    
    
    imshow(img)
    colormap(gray)
    subplot(1,2,2)
    img_mask = imread([pwd, '\train\', img_masks{img_idx}]);
    img1_masked = img.*(img_mask./255);
    imshow(img1_masked)
    colormap(gray)
    pause(0.5)
end





img_mask = imresize(img_mask,0.2);
X_cm = sum(sum(meshgrid(1:size(img_mask,1),zeros(1,size(img_mask,2)))'.*img_mask))./sum(img_mask(:))
Y_cm = sum(sum(meshgrid(1:size(img_mask,2),zeros(1,size(img_mask,1))).*img_mask))./sum(img_mask(:))









