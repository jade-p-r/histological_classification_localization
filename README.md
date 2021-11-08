# Submission for Primaa Technical Interview - Jade Perdereau

This repository gathers my work for the two tasks given for the technical interview at Primaa.


## Installation
If needed, create a custom virtual environment 

Clone the repo using git clone 

cd into current directory

Install the required packages using pip3 install requirements.txt

To run torchvision for nuclei detection, you also need to install pycocotools, for example via : pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI

The trained models for both tasks have been sent to you via google drive. The associated data also. please unzip it as the following arborescence :

- classification/train/nuclei/001.png

- localization/train/001.png

## Input data
The given data separates data in train and test sets, the test set being 10% of the size of the training set. We perform validation on a subset from the training set of 10%
## Nuclei Images Classification
The following assumption has been made : there are some mistakes in image labelling, notably in the test set (ie no_nuclei_120.png).
Some images were also duplicated in the training set and were removed. We computed various indicators to train a machine learning model - Support Vector Classifier. We spent more effort on feature design than model tuning


Create an output dir via :

cd ./classification

mkdir output

Unzip model in the output folder
### Training
To train the model, run :
(if not in classif directory) cd classification
python3 train.py

### Inference
To run predictions on the test set, run:
(if not in classif directory) cd classification
python3 predict.py

Inference can display images which were wrongly classified, namely false positives and false negatives.
We chose to assess precision and recall as performance metrics.
Our model achieves a precision of 89% and a recall of 93% / 95% if we take into consideration the labelling "errors".

## Nuclei detection

To detect nuclei, we made the following assumptions : images with no bounding boxes/ nuclei were removed from the training set.
Some labelling mistakes were also present notably in the test set (boxes equal to whole image) and they were not considered in the assessment.
We used transfer learning with FastRCNN for its accuracy and changed the maximum detection threshold given that some images contain a large number of nuclei.

The model was trained on 5 epochs with batch size 1 

From the current directory, 
### Training
cd localization
To train the model run:

python3 train_pytorch.py
### Inference
We chose to also display precision, recall and F1 score on the test set to assess our model's performance.
To run prediction metrics on the test , run:
python3 predict.py --input=test

Our model achieves a Precision of 72%, Recall of 58% and F1-score of 64% on the test set.

A few examples of detection below :

![alt text](https://github.com/jade-p-r/primaa_submission/blob/main/screens/Output_screenshot_08.11.2021.png?raw=true)


![alt text](https://github.com/jade-p-r/primaa_submission/blob/main/screens/Output_screenshot_08.11.2021_1.png?raw=true)


![alt text](https://github.com/jade-p-r/primaa_submission/blob/main/screens/Output_screenshot_08.11.2021_2.png?raw=true)


![alt text](https://github.com/jade-p-r/primaa_submission/blob/main/screens/Output_screenshot_08.11.2021_3.png?raw=true)