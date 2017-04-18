% The main file for handling and analyzing the data:


%% Make a list of all images in the training files
% a = dir('train');
% imgs = {};
% img_masks = {};
% for i = 3:length(a)
%     name = a(i).name;
%     if findstr('tif',name)
%     if findstr('mask',name)
%         %img_masks{end+1} = name;
%     else
%         imgs{end+1} = name;
%         img_masks{end+1} = [name(1:end-4),'_mask.tif'];
%     end
%     end
% end

mode = 'train'
mode = 'test'
size = [80, 112]

a = importdata([mode,'\\names.txt']);
imgs = {};
img_masks = {};
for i = 1:length(a)
    try name = a{i};
    catch name = a(i);
    end
    if isnumeric(name)
        name = num2str(name);
    end
    imgs{end+1} = [name, '.tif'];
    img_masks{end+1} = [name,'_mask.tif'];
end


%% Redo with name list:


%% Plotting the data:
%PlotDifferentData

%% Analyzing Masks:
%MaskStats

%% Using the information from these masks to make outputs for the NN:
if strcmp(mode,'train')
    Crop_Images_for_NN(imgs,img_masks,0.2,2)
elseif strcmp(mode,'test')
end
Scale_Filter_Save_Full(imgs,img_masks,mode,size)

    