function Crop_Images_for_NN(imgs, img_masks, SCALE, ext)
% Takes all images and saves cropped versions of the mask area and 
% many other points in the surroundings:


%% First load all masks into memory and convert to binary:
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

%% Analyze the masks:
% The X and Y length that captures 80% of the mask before rescaling.
X_len = zeros(1,length(mask_idxs)); xmax = size(img_mask,1);
Y_len = zeros(1,length(mask_idxs)); ymax = size(img_mask,2);
Areas = zeros(1,length(mask_idxs));
X_cm = zeros(1,length(mask_idxs));
Y_cm = zeros(1,length(mask_idxs));
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
end

% Make the width and height large enough to cover 80% of the masks X and Y
cutoff = 0.8;
XLEN = round(fminsearch( @(cut) abs(sum(X_len<cut)./sum(X_len>0)-cutoff),100))
YLEN = round(fminsearch( @(cut) abs(sum(Y_len<cut)./sum(Y_len>0)-cutoff),100))

% Clean up masks - don't need them anymore!
clear IMG_MASKS

%% Start working with the data:

% Data for the images at the scale given:
N_samples = length(imgs);
xhspan = round(XLEN*SCALE/2);yhspan = round(YLEN*SCALE/2);
xspan = xhspan*2+1; yspan = yhspan*2+1; pix = xspan*yspan;
X_cm_sc = round(X_cm.*SCALE); Y_cm_sc = round(Y_cm.*SCALE);

% Set filter parameters for Beltrami Filter:
addpath([pwd '\BeltramiPD']); 
beta = 1; tol = 1e-3; lambda = 1/10;
r1 = 0.1; r2 = 0.1/lambda; Maxit = 2000;

% Set random seed for reproducibility:
rng(2016);

% Load each image. 
% If masked, create new image with mask at its X_cm, Y_cm;
% If not masked, create new imahe with mask at ranom X_cm, Y_cm;
mid = ext+1; tot = 2*ext+1;
IMGs = zeros(N_samples,tot,tot,pix,'uint8'); 
Targets = false(N_samples,1); 
Ocomp = 0; 
c_idxs = zeros([tot,tot]);

for img_idx = 1:N_samples
    % Load the data:
    img = imread( strcat(pwd, ['\train\'], imgs{img_idx}));
    % Rotate, filter, and scale the data:
    img = double(img); img = 256*(img-min(img(:)))/(max(img(:))-min(img(:)));
    [img,k] = BeltramiPD( img, beta, lambda, r1, r2, tol, Maxit );
    %img = imrotate(img,THETA);
    img = imresize(img,SCALE);
    img = uint8(img);
    xmax = size(img,1); ymax = size(img,2);
    % Mask time:
    [mask,mask_idx] = find(mask_idxs==img_idx);
    if mask
        xs = [X_cm_sc(mask_idx)-xhspan:X_cm_sc(mask_idx)+xhspan];
        ys = [Y_cm_sc(mask_idx)-yhspan:Y_cm_sc(mask_idx)+yhspan];
    else
        mask = 0; mask_idx = randi(length(mask_idxs),1);
        xs = [X_cm_sc(mask_idx)-xhspan:X_cm_sc(mask_idx)+xhspan];
        ys = [Y_cm_sc(mask_idx)-yhspan:Y_cm_sc(mask_idx)+yhspan];
    end
    if min(xs)<1, xs=xs-(min(xs)-1);elseif max(xs)>xmax, xs=xs-(max(xs)-xmax); end
    if min(ys)<1, ys=ys-(min(ys)-1);elseif max(ys)>ymax, ys=ys-(max(ys)-ymax); end
    img_main = img(xs,ys);
    % Add data to image matrix:
    IMGs(img_idx,mid,mid,:) = reshape(img_main,[1,pix]);
    Targets(img_idx) = logical(mask);
    c_idxs(mid,mid) = c_idxs(mid,mid) +1;
    

    % Iterate around the mask:
    for j = 1:tot
        for k = 1:tot
            if and(j==mid,k==mid), continue, end
            xsn = xs+(j-mid)*xspan; ysn = ys+(k-mid)*yspan;
            if or(min(xsn)<1, max(xsn)>xmax),continue,
            elseif or(min(ysn)<1, max(ysn)>ymax), continue,
            else
                c_idxs(j,k) = c_idxs(j,k) + 1;
                img_now = img(xsn,ysn);
                IMGs(img_idx,j,k,:) = reshape(img_now,[1,pix]);
            end
        end
    end

    comp = round(img_idx./N_samples*100);
    if comp>Ocomp
        Ocomp = comp;
        display([num2str(comp) '% Complete'])
    end
end
display( [num2str(round(sum(Targets)./length(Targets)*100)) '% of targets are BP'] )

folder = ['\Neural_Network_Crops\']

for i = 1:tot
    for j = 1:tot
        csvwrite([pwd folder 'Imgs_' num2str(i) num2str(j) '_' num2str(xspan) 'x' num2str(yspan) '.csv'],IMGs(1:c_idxs(i,j),i,j,:))
    end
end
csvwrite([pwd folder 'Imgs_Targets.csv'],Targets)




end