%The first attempt at setting up the data for a convolutional neural
%network. The general pipeline is as follows:
% 1) Determine the optimum box size (and rotation?) 
% 2) Filter image.
% 3) Downscale the image.
% 4) Cut example positive and negative instances
% 5) Save instances and labels.


%% Creating the data sets based on the parameters determined in "MaskStats"
% See MaskStats. Optimum stats are here:
% For rotated data:
ROT = 0; if ROT, load('XY_rot_cm.mat'), 
else, load('XY_cm.mat'), end
%loads X_cm, Y_cm, XLEN = 140; YLEN = 96; THETA = 22; mask_idxs
SCALE = 0.20;
N_samples = length(imgs);
xhspan = round(XLEN*SCALE/2);yhspan = round(YLEN*SCALE/2); pix = (xhspan*2+1)*(yhspan*2+1);
X_cm_sc = round(X_cm.*SCALE); Y_cm_sc = round(Y_cm.*SCALE);
% Set filter parameters:
addpath([pwd '\BeltramiPD']); 
beta = 1; tol = 1e-3; lambda = 1/10;
r1 = 0.1; r2 = 0.1/lambda; Maxit = 2000;
% Set random seed for reproducibility:
rng(2016);
% Load each image. 
% If masked, create new image with mask at its X_cm, Y_cm;
% If not masked, create new imahe with mask at ranom X_cm, Y_cm;
testing = 0; if testing, N_samples = 100; else
IMGs = zeros(N_samples,pix,'uint8'); Targets = false(N_samples,1); Ocomp = 0; end
for img_idx = 1:N_samples
    % Load the data:
    img = imread( strcat(pwd, '\train\', imgs{img_idx}));
    % Rotate, filter, and scale the data:
    img = double(img); img = 256*(img-min(img(:)))/(max(img(:))-min(img(:)));
    [img,k] = BeltramiPD( img, beta, lambda, r1, r2, tol, Maxit );
    img = imrotate(img,THETA);
    img = imresize(img,SCALE);
    img = uint8(img);
    xmax = size(img,1); ymax = size(img,2);
    % Mask time:
    [mask,mask_idx] = find(mask_idxs==img_idx);
    if mask
        xs = [X_cm_sc(mask_idx)-xhspan:X_cm_sc(mask_idx)+xhspan];
        ys = [Y_cm_sc(mask_idx)-yhspan:Y_cm_sc(mask_idx)+yhspan];
    else
        mask = 0;
        mask_idx = randi(length(mask_idxs),1);
        xs = [X_cm_sc(mask_idx)-xhspan:X_cm_sc(mask_idx)+xhspan];
        ys = [Y_cm_sc(mask_idx)-yhspan:Y_cm_sc(mask_idx)+yhspan];
    end
    if min(xs)<1, xs=xs-(min(xs)-1);elseif max(xs)>xmax, xs=xs-(max(xs)-xmax); end
    if min(ys)<1, ys=ys-(min(ys)-1);elseif max(ys)>ymax, ys=ys-(max(ys)-ymax); end
    img = img(xs,ys);
    
    if testing
        imagesc(img)
        title(['Mask: ' num2str(mask)])
        colormap(gray)
        pause(1)
    else
        IMGs(img_idx,:) = reshape(img,[1,pix]);
        Targets(img_idx) = logical(mask);
        comp = round(img_idx./N_samples*100);
        if comp>Ocomp
            Ocomp = comp;
            display([num2str(comp) '% Complete'])
        end
    end
end
display( [num2str(round(sum(Targets)./length(Targets)*100)) '% of targets are BP'] )

csvwrite(['Imgs_' num2str(xhspan*2+1) 'x' num2str(yhspan*2+1) '.csv'],IMGs)
csvwrite(['Imgs_Targets.csv'],Targets)


%% Alternatively, try full images with/without rotation. See what a NN can do.
SCALE = 0.20;
%SCALE = [80, 112]
ROT = 0; if ROT, load('XY_rot_cm.mat'), end
N_samples = length(imgs);
% Set filter parameters:
addpath([pwd '\BeltramiPD']); 
beta = 1; tol = 1e-3; lambda = 1/10;
r1 = 0.1; r2 = 0.1/lambda; Maxit = 2000;
% Set random seed for reproducibility:
rng(2016);
% Load each image and scale appropriately. The new number of pixels is...
try mode
catch mode = 'train'
end
img = imread( strcat(pwd, ['\' mode '\'], imgs{1}));
if ROT, img = imrotate(img,THETA); end
img = imresize(img,SCALE);
xscale = size(img,1); yscale = size(img,2);
pix = xscale.*yscale;
IMGs = zeros(N_samples,pix,'uint8');  
MASKs = zeros(N_samples,pix,'uint8');
Ocomp = 0;
for img_idx = 1:N_samples
    % Load the data:
    img = imread( strcat(pwd,  ['\' mode '\'], imgs{img_idx}));
    % Rotate, filter, and scale the data:
    img = double(img); img = 256*(img-min(img(:)))/(max(img(:))-min(img(:)));
    [img,k] = BeltramiPD( img, beta, lambda, r1, r2, tol, Maxit );
    if ROT, img = imrotate(img,THETA); end
    img = imresize(img,SCALE);
    img = uint8(img);
    
    if strcmp(mode,'train')
    mask = imread( strcat(pwd,  ['\' mode '\'], img_masks{img_idx}));
    mask = imresize(mask,SCALE);
    mask = mask>0.5;
    MASKs(img_idx,:) = reshape(mask,[1,pix]);
    end
    
    IMGs(img_idx,:) = reshape(img,[1,pix]);
    comp = round(img_idx./N_samples*100);
    if comp>Ocomp
        Ocomp = comp;
        display([num2str(comp) '% Complete'])
    end
end

if ROT == 0
    csvwrite(['Full_Imgs_' mode '_' num2str(xscale) 'x' num2str(yscale) '.csv'],IMGs)
    if strcmp(mode,'train')
    csvwrite(['Full_Masks_'  mode '_' num2str(xscale) 'x' num2str(yscale) '.csv'],MASKs)
    end
elseif ROT == 1
    csvwrite(['Full_Imgs_Rot'  mode '_' num2str(xscale) 'x' num2str(yscale) '.csv'],IMGs)
end


%% Adding in more false data from other parts of the data sets:
ROT = 0; if ROT, load('XY_rot_cm.mat'), 
else, load('XY_cm.mat'), end
%loads X_cm, Y_cm, XLEN = 140; YLEN = 96; THETA = 22; mask_idxs
SCALE = 0.20;
N_samples = length(imgs);
xhspan = round(XLEN*SCALE/2);yhspan = round(YLEN*SCALE/2);
xspan = xhspan*2+1; yspan = yhspan*2+1; pix = xspan*yspan;
X_cm_sc = round(X_cm.*SCALE); Y_cm_sc = round(Y_cm.*SCALE);
% Set filter parameters:
addpath([pwd '\BeltramiPD']); 
beta = 1; tol = 1e-3; lambda = 1/10;
r1 = 0.1; r2 = 0.1/lambda; Maxit = 2000;
% Set random seed for reproducibility:
rng(2016);
% Load each image. 
% If masked, create new image with mask at its X_cm, Y_cm;
% If not masked, create new imahe with mask at ranom X_cm, Y_cm;
IMGs = zeros(N_samples,3,3,pix,'uint8'); Targets = false(N_samples,1); 
Ocomp = 0; c_idxs = zeros([3,3]);
% Outside of the main 3x3 grid:
ex_i = [3,4,4]; ex_j = [4,3,4]; ex_c_idxs = zeros([1, length(ex_i)]);
EX_IMGs =  zeros(N_samples,length(ex_i),pix,'uint8');
for img_idx = 1:N_samples
    % Load the data:
    img = imread( strcat(pwd, '\train\', imgs{img_idx}));
    % Rotate, filter, and scale the data:
    img = double(img); img = 256*(img-min(img(:)))/(max(img(:))-min(img(:)));
    [img,k] = BeltramiPD( img, beta, lambda, r1, r2, tol, Maxit );
    img = imrotate(img,THETA);
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
    IMGs(img_idx,2,2,:) = reshape(img_main,[1,pix]);
    Targets(img_idx) = logical(mask);
    c_idxs(2,2) = c_idxs(2,2) +1;
    

%     % Iterate around the mask:
%     for j = 1:3
%         for k = 1:3
%             if and(j==2,k==2), continue, end
%             xsn = xs+(j-2)*xspan; ysn = ys+(k-2)*yspan;
%             if or(min(xsn)<1, max(xsn)>xmax),continue,
%             elseif or(min(ysn)<1, max(ysn)>ymax), continue,
%             else
%                 c_idxs(j,k) = c_idxs(j,k) + 1;
%                 img_now = img(xsn,ysn);
%                 IMGs(img_idx,j,k,:) = reshape(img_now,[1,pix]);
%             end
%         end
%     end
    
    % Take care of any extras:
    % Iterate around the mask:
    for ex_idx = 1:length(ex_i)
        j = ex_i(ex_idx);
        k = ex_j(ex_idx);
        xsn = xs+(j-2)*xspan; ysn = ys+(k-2)*yspan;
        if or(min(xsn)<1, max(xsn)>xmax),continue,
        elseif or(min(ysn)<1, max(ysn)>ymax), continue,
        else
            ex_c_idxs(ex_idx) = ex_c_idxs(ex_idx) + 1;
            img_now = img(xsn,ysn);
            EX_IMGs(ex_c_idxs(ex_idx),ex_idx,:) = reshape(img_now,[1,pix]);
        end
    end

    comp = round(img_idx./N_samples*100);
    if comp>Ocomp
        Ocomp = comp;
        display([num2str(comp) '% Complete'])
    end
end
display( [num2str(round(sum(Targets)./length(Targets)*100)) '% of targets are BP'] )

for i = 1:3
    for j = 1:3
        csvwrite(['Imgs_' num2str(i), num2str(j) '_' num2str(xspan) 'x' num2str(yspan) '.csv'],IMGs(1:c_idxs(i,j),i,j,:))
    end
end
csvwrite('Imgs_Targets.csv',Targets)


% for ex_idx = 1:length(ex_i)
%     csvwrite(['Imgs_EX_' num2str(ex_idx) '_' num2str(xspan) 'x' num2str(yspan) '.csv'],EX_IMGs(1:ex_c_idxs(ex_i),:))
% end
ex_i = 1
csvwrite(['Imgs_EX_' num2str(ex_idx) '_' num2str(xspan) 'x' num2str(yspan) '.csv'],EX_IMGs(1:ex_c_idxs(ex_i),ex_i,:))







%% Process train data in the same way:
% Make a list of all images in the training files







