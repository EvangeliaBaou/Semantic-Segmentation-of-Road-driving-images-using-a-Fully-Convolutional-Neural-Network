# Semantic Segmentation of Road/Driving images using a Fully Convolutional Neural Network
This notebook illustrates how to build a Fully Convolutional Neural Network for semantic image segmentation.
The model will be trained on a dataset of images and labeled semantic segmentations captured via CARLA self-driving car simulator. The dataset is hosted in a Google bucket so you will need to download it first and unzip to a local directory.
A pretrained VGG-16 network will be used for the feature extraction path, then followed by an FCN-8 network for upsampling and generating the predictions. 
The output will be a label map (i.e. segmentation mask) with predictions for 23 classes. All code was implemented in Google Colab.

## Requirements
* os for interacting with the operating system
* zipfile for dataset extraction
* numpy for data manipulation
* tensorflow for building the model 
* matplotlib for visualization
* tensorflow_datasets (???)
* seaborn
## Dataset
The dataset you just downloaded contains folders for images and annotations. The images contain the original images while the annotations contain the pixel-wise label maps. Each label map has the shape (height, width , 1) with each point in this space denoting the corresponding pixel's class. Classes are in the range [0, 21] (i.e. 22 classes) and the pixel labels correspond to these classes:
| Value         | Class Name |
| ------------- | ------------- |
| 0             | Unlabeled  |
| 1             | Building  |
| 2             | Fence  |
| 3             | Other  |
| 4             | Pedestrian  |
| 5             | Pole  |
| 6             | RoadLine  |
| 7             | Road  |
| 8             | SideWalk  |
| 9             | Vegetation  |
| 10            | Vehicles  |
| 11            | Wall  |
| 12            | TrafficSign  |
| 13            | Sky  |
| 14            | Ground  |
| 15            | Bridge  |
| 16            | RailTrack  |
| 17            | GuardRail  |
| 18            | TrafficLight  |
| 19            | Static  |
| 20            | Dynamic  |
| 21            | Water  |
| 22            | Terrain  |

For example, if a pixel is part of road, then that point will be labeled 7 in the label map.

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

# Model
AS mentioned earlier, a VGG-16 network will be used for the encoder and FCN-8 for the decoder. This is the diagram of the model's architecture:

![alt text](https://github.com/LiaBaou/Semantic-Segmentation-of-Road-driving-images-using-a-Fully-Convolutional-Neural-Network/blob/main/fcn8.png)

## Define Pooling Block of VGG
VGG networks have repeating blocks thus a function was created to summarize this process. Each block has convolutional layers followed by a max pooling layer which downsamples the image.

## Define VGG-16

The encoder is built as shown below.

   1. Create 5 blocks with increasing number of filters at each stage.
   2. The number of convolutions, filters, kernel size, activation, pool size and pool stride will remain constant.
   3. Load the pretrained weights after creating the VGG 16 network.[The pretrained weights for VGG 16 can be found here: 
      https://github.com/fchollet/deep-learning-models/releases/tag/v0.1]
   5. Additional convolution layers will be appended to extract more features.
   6. The output will contain the output of the last layer and the previous four convolution blocks.
## Define FCN 8 Decoder
## Define final model
## Compile the model
## Train the model
## Evaluate the model
The intersection-over-union and the dice score are used as metrics to evaluate the model. In particular:
* intersection-over-union: is known to be a good metric for measuring overlap between two bounding boxes or masks. If the prediction is completely correct, IoU = 1. The lower the IoU, the worse the prediction result.
![alt text](https://miro.medium.com/max/3000/1*kK0G-BmCqigHrc1rXs7tYQ.jpeg)
