function Scale_Filter_Save_Full(imgs,img_masks,mode,SCALE)
%% Save full images after filer and scale. 

% Set filter parameters and other basic parameters:
N_samples = length(imgs);
addpath([pwd '\BeltramiPD']); 
beta = 1; tol = 1e-3; lambda = 1/10;
r1 = 0.1; r2 = 0.1/lambda; Maxit = 2000;

% Set folder:
folder = '\Neural_Network_Full\';

% Set random seed for reproducibility:
rng(2016);


% Load each image and scale appropriately. The new number of pixels is...
img = imread( strcat(pwd, ['\' mode '\'], imgs{1}));
img = imresize(img,SCALE);
xscale = size(img,1); yscale = size(img,2);
pix = xscale.*yscale;
IMGs = zeros(N_samples,pix,'uint8');  
if strcmp(mode,'train')
MASKs = zeros(N_samples,pix,'uint8');
end
Ocomp = 0;
for img_idx = 1:N_samples
    % Load the data:
    img = imread( strcat(pwd,  ['\' mode '\'], imgs{img_idx}));
    % Rotate, filter, and scale the data:
    img = double(img); img = 256*(img-min(img(:)))/(max(img(:))-min(img(:)));
    [img,k] = BeltramiPD( img, beta, lambda, r1, r2, tol, Maxit );
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

% Save the results:
csvwrite([pwd folder 'Full_Imgs_' mode '_' num2str(xscale) 'x' num2str(yscale) '.csv'],IMGs)
if strcmp(mode,'train')
csvwrite([pwd folder 'Full_Masks_'  mode '_' num2str(xscale) 'x' num2str(yscale) '.csv'],MASKs)
end










end