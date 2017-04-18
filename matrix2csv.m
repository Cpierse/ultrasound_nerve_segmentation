function [ ] = matrix2csv( img_mat, name )
%MATRIX2CSV Converts row by col by N image matrix into an unrolled csv.
% The csv will be of the shape N by rows*cols.

[rows,cols,N]  = size(img_mat);
img_mat = reshape(img_mat,[rows*cols,N]);
img_mat = transpose(img_mat);


csvwrite([name '_' num2str(rows) 'x' num2str(cols) '.csv'],img_mat)
display([ char(name) '_' num2str(rows) 'x' num2str(cols) '.csv saved'])
end

