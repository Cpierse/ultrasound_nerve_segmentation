%Applies a low pass gaussian filter to an image, with a standard deviation
%st (low pass is more severe for st -> 0)
% Credit: Conrad Foo
function img = lowpassIm(img,st)
    %Create gaussian filter
    lpfilt = fspecial('gaussian',[size(img,1),size(img,2)],st);
    %FFT of image
    z = fft2(img);
    z = fftshift(z);
    %Apply filter
    z = z.*lpfilt;
    %Invert FFT
    z = ifftshift(z);
    img = ifft2(z,'symmetric');
end
%st = 4 for us right now