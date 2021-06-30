# Semantic-Segmentation-of-Road-driving-images-using-a-Fully-Convolutional-Neural-Network
This notebook illustrates how to build a Fully Convolutional Neural Network for semantic image segmentation.
The model will be trained on a dataset of images and labeled semantic segmentations captured via CARLA self-driving car simulator. The dataset is hosted in a Google bucket so you will need to download it first and unzip to a local directory.
A pretrained VGG-16 network will be used for the feature extraction path, then followed by an FCN-8 network for upsampling and generating the predictions. 
The output will be a label map (i.e. segmentation mask) with predictions for 23 classes.

## Requirements
* os for interacting with the operating system
* zipfile for dataset extraction
* numpy for data manipulation
* tensorflow for building the model 
* matplotlib for visualization
* tensorflow_datasets (???)
* seaborn

## Data Preparation/exploration
1. Resizing the height and width of the input images and label maps (224 x 224px by default)
2. Normalizing the input images' pixel values to fall in the range [-1, 1]
3. Reshaping the label maps from (height, width, 1) to (height, width, 22) with each slice along the third axis having 1 if it belongs to the class corresponding to that slice's index else 0. For example, if a pixel is part of a tree, then using the table above, that point at slice #1 will be labeled 1 and it will be 0 in all other slices. 
For example : <br/>
If we have a label map with 4 classes. <br/>
n_classes = 4 <br/>
And this is the original annotation. <br/>
orig_anno = [0 1 2 3] <br/>
Then the reshaped annotation will have 4 slices and its contents will look like this: <br/>
reshaped_anno = [1 0 0 0] [0 1 0 0] [0 0 1 0] [0 0 0 1]
