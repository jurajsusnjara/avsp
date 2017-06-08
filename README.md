# AVSP Project

This project was part of the course "Analysis of Massive Datasets" at Faculty of Electrical Engineering and Computing, Zagreb and was coordinated by doc. dr. sc. Marin Šilić.

Team working on project:
- Juraj Šušnjara
- Damjan Miko
- Domagoj Krivošić
- Luka Novak

Project included analysing paper on "Deep Feature Interpolation for Image Content Changes" (Upchurch, Gardner, Bala, Pless, Snavely, Weinberger) which is given at https://arxiv.org/pdf/1611.05507.pdf and reproducing results. Implementation given at https://github.com/dsanno/chainer-dfi.

This repository contains python scripts for making PCA analysis on deep feature space of images. How to download and use VGG19 neural network used to transform images in deep feature space can be found at https://github.com/dsanno/chainer-dfi.

**df_image.py**

Contains class and methods that process image, extract deep features and transform image from deep feature back to pixel space. This class and methods are used in following scripts.

**pca_train.py**

Uses IncrementalPCA from sklearn to train PCA model which is than dumped to file. Example of usage:

`python pca_train.py -batch_size 300 -n_components 300 -rootdir "lfwdeepfunneled" -max_samples 3000 -model_dump "model.pkl"`

Images used for training should be aligned and same dimensions. In this project LFW deep funneled images are used that can be downloaded from http://vis-www.cs.umass.edu/lfw.

**pca_test.py**

Loads previously trained PCA model, loads given image, transforms it into deep feature space (~1000000 dimensions vector), reduce dimensions using loaded PCA model, makes inverse transform to restore dimensions, transforms image back to pixel space and save image to file. In this way produced and original image can be compared to see performance of PCA model. Example of usage:

`python pca_test.py -input "input_image.jpg" -output "output_image.jpg" -model_dump "model.pkl"`

Resulting images can be found in results folder. Eg. if the name of the file is: result_100samples_100dim.jpg it means PCA was trained on 100 images and will reduce images to 100 dimensions.
