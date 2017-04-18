%Credit: Conrad Foo
function [stack,id] = createStack(pnumber,keys,c)
    disp('Reading from disk..');
    pattern = ['^' num2str(pnumber) '_([0-9]+)_mask.tif'];
    stack = [];
    idx = [];
    for i = 1:length(keys)
        z = regexp(keys{i},pattern,'tokens');
        if ~isempty(z)
            imname = c(keys{i});
            stack = cat(3,stack,imread(imname{1}));
            idx = cat(1,idx,i);
        end
    end
    
    disp('Calculating cost matrix..');
    distmat = zeros(size(stack,3));
    unvisited = ones(size(stack,3),1);
    maxd = 0;
    for i = 1:size(stack,3)
        for j = 1:i-1
            u = single(reshape(stack(:,:,i),size(stack,1)*size(stack,2),1));
            v = single(reshape(stack(:,:,j),size(stack,1)*size(stack,2),1));
            distmat(i,j) = dot(u-v,u-v);
            distmat(j,i) = distmat(i,j);
            if distmat(i,j) > maxd
                maxd = distmat(i,j);
                maxdistPair = [i,j];
            end
        end
        distmat(i,i) = 1e999;
    end
    
    disp('Reordering stack..');
    tr = zeros(size(stack,3),1);
    tr(1) = maxdistPair(1);
    unvisited(maxdistPair(1)) = 1e999;
    for i = 2:size(stack,3)
        id = find(distmat(tr(i-1),:).*unvisited' == min(distmat(tr(i-1),:).*unvisited'));
        unvisited(id(1)) = 1e999;
        tr(i) = id(1);
    end
    stack2 = zeros(size(stack),'like',stack);
    id2 = zeros(size(idx),'like',idx);
    for i = 1:size(stack,3)
        stack2(:,:,i) = stack(:,:,tr(i));
        id2(i) = idx(tr(i));
    end
    stack = stack2;
    id = id2;
end