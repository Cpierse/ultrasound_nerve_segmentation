%Overlap between the best elliptical fit and the actual region.
%Credit: Conrad Foo
function z = ellipticity(labelImg)
    %Mask for the elliptical region
    emask = false(size(labelImg));
    
    %Get the elliptical fit for each region.
    z = regionprops(labelImg,'Centroid','MajorAxisLength','MinorAxisLength','Orientation','Area');
    labelImg = bsxfun(@gt,labelImg,0); % All elements greater than 1
    
    for i = 1:length(z)
        %Grab ellipse properties.
        emask(:,:) = 0;
        x0 = round(z(i).Centroid(1));
        y0 = round(z(i).Centroid(2));
        a = z(i).MajorAxisLength/2;
        b = z(i).MinorAxisLength/2;
        angle = z(i).Orientation;
        if ~isnan(x0) || ~isnan(y0)    
            %Calculate elliptical area with no rotation.
            for x = 0:round(a)
                xidx = round(min(size(emask,2)/2+x,size(emask,2)));
                yext = round(b*sqrt(1-x^2/a^2));
                yidx = round(size(emask,1)/2-yext):round(size(emask,1)/2+yext);
                yidx = yidx(yidx > 0 & yidx <= size(emask,1));
                emask(yidx,xidx) = 1;
                xidx = round(max(1,size(emask,2)/2-x));
                emask(yidx,xidx) = 1;
            end
            
            %Rotate ellipse by angle and translate to the correct place.
            emask = imrotate(emask,angle,'crop');
            emask = imtranslate(emask,[-size(emask,2)/2+x0,-size(emask,1)/2+y0],'OutputView','same');
            
            %Stats of fit
            numPix = sum(sum(emask.*labelImg,2),1);
            %Precision
            p = numPix/sum(emask(:));
            %Recall
            r = numPix/z(i).Area;
            %F-score
            z(i).ellipticity = 2*p*r/(p+r);
        end
    end
end