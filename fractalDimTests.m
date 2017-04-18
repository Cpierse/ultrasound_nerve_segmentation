function fractalDimTests(img,img_mask, nlevels)
%Tests the concept of using fractal dimensions, variance, and other 
%properties to identify the region of
%interest. Uses the boxcounting method with nlevel thresholds. The input
%image is suggested to be gaussian filtered. 

% Delete these:
nlevels = 20;
% First load an image where there is a mask:
img_idx = 1
img_idx = 3
img = imread( strcat(cdir, '\train\', imgs{img_idx}));
img_mask = imread([cdir, '\train\', img_masks{img_idx}]);
st = 15;
img = lowpassIm(img,st);
% Delete end

% First get a stack of bw thresholds and plot them with mask:
bStack = ttbd(img,nlevels,1);
bStackOne = zeros(size(bStack,1),size(bStack,2),1,class(img));
for i = 1:size(bStack,3); bStackOne = bStackOne+bStack(:,:,i).*i; end
highlightBP(bStackOne,img_mask);

% Define a box that contains the mask
[rows,cols] = find(img_mask);
x0 = min(rows); x1 = max(rows); x_len = x1-x0;
y0 = min(cols); y1 = max(cols); y_len = y1-y0;
b_slice = bStackOne(x0:x1,y0:y1)
imagesc(b_slice)

% Let's try a scan of half size boxes:
overlap = 5;
x_scans = floor(size(img,1)./x_len*overlap);
y_scans = floor(size(img,1)./y_len*overlap);
boxes = zeros(x_scans*(y_scans),4);
n_box = size(boxes,1)
for i = 1:x_scans
    for j = 1:y_scans
        boxes((i-1)*y_scans + j,:) = [(i-1).*round(x_len/overlap)+1, i*round(x_len/overlap), (j-1).*round(y_len/overlap)+1, j*round(y_len/overlap)];
    end
end

Means = zeros(n_box,1);
Vars = zeros(n_box,1);
UniRat = zeros(n_box,1);
FracDs = zeros(n_box,nlevels);

for it = 1:x_scans*y_scans
    o_slice = bStackOne(boxes(it,1):boxes(it,2),boxes(it,3):boxes(it,4));
    imagesc(o_slice)
    
    overlap = sum(sum(img_mask(boxes(it,1):boxes(it,2),boxes(it,3):boxes(it,4))))./sum(img_mask(:));
    title(['Overlap = ' num2str(overlap)])
    
    pause(0.5)

    for i = 1:nlevels
        [n, r] = boxcount(o_slice==6,'slope');
        a = polyfit(log10(r(r<100)),log10(n(r<100)),1);
        FracDs(it,i) = -a(1);
    end
    
    
    Means(it) = mean(o_slice(:));
    Vars(it) = var(o_slice(:));
    UniRat(it) = length(unique(o_slice(:)))./nlevels;
    Overlap(it) = overlap;
end

FracDs(Overlap>0,:)



end