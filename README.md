# ultrasound_nerve_segmentation
Code for our Kaggle entries in the Ultrasound Nerve Segmentation competition

## Overview:
Our algorithm for predicting which regions of an ultrasound image contain the brachial plexus nerves. A large portion of the data was processed using Matlab, while our predictive algorithms were trained in Python. 

### Prerequisites:
- [Data](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data) from the Kaggle competition
- [Boxcounting](https://www.mathworks.com/matlabcentral/fileexchange/13063-boxcount?focused=5083236&tab=function)  algorithm for fractal analysis
- [Otsu's thresholding](https://www.mathworks.com/matlabcentral/fileexchange/26532-image-segmentation-using-otsu-thresholding) algorithm for image segmentation
- [Beltrami filter](https://www.mathworks.com/matlabcentral/fileexchange/47470-efficient-beltrami-image-denoising-and-deconvolution) for image denoising
- [Rotate around](https://www.mathworks.com/matlabcentral/fileexchange/40469-rotate-an-image-around-a-point) function for easy rotations
- [Saliency map](https://github.com/mayoyamasaki/saliency-map) algorithm for highlight potential ROIs
- Python packages: Keras, Theano, OpenCV