# Face-Mask-Detection

This project is developed to study popular deep learning networks in image classification and to use transfer learning in a practical application.  

## Objectives

1. Develop a neural network classifier to identify the eight classes defined by gender and different face mask worn type combinations.
1. Tune the Keras/Tensorflow based model to obtain 90 % test accuracy with a balanced dataset.

## License

The source code hosted in this repository is shared under MIT license.

## Sponsor

DataDisca Pty Ltd, Melbourne, Australia

[https://www.datadisca.com](https://www.datadisca.com)

## Dataset

Publicly available below Kaggle datasets were used for training, testing and validation. Please check the respective licenses of the datasets mentioned here before using them.

Note: The "Type ID" descriptions mentioned in the datasets should be corrected as in [this](https://www.kaggle.com/tapakah68/medical-masks-part1/discussion/254996) discussion post.

1. [500 GB of images with people wearing masks. Part 1](https://www.kaggle.com/tapakah68/medical-masks-part1/) - Dataset 1 for training and testing
2. [500 GB of images with people wearing masks. Part 7](https://www.kaggle.com/tapakah68/medical-masks-part7) - Dataset 2 for validation

## Prerequisites

Latest tested versions are mentioned inside the brackets along with the library names for reference.

1. Python (3.9.7)
2. Jupyter Notebook (6.4.6) with IPython (7.29.0)
3. Pillow (8.4.0)
4. Numba (0.54.0rc1)
5. Numpy (1.22.0)
6. Pandas (1.3.4)
7. OpenCV (4.5.1)
8. Matplotlib  (3.5.0)
9. Tensorflow (2.9.0.dev20220102) including Keras (2.9.0.dev2022010308) and Tensorboard (2.8.0a20220102)


## Preprocessing

The raw dataset is preprocessed with [preprocess.ipynb](Preprocess/preprocess.ipynb) in order to remove non-image files and to categorize into eight classes defined by gender and different face mask worn type combinations by checking the filenames.     

Before running the preprocessing script, the raw dataset should be extracted to "original_images".    
After runnning this script, there will be subfolders with classes mentioned in "classify_names" inside the "temp_base" preprocessed dataset output directory.

## Detection Testing

A sample script, [check_image.ipynb](Test/check_image.ipynb) is provided to test a single image from a local file or an URL with the trained model.  
The output would the classification from "classify_names" and a general description from "classify_desc" along with the image display.

