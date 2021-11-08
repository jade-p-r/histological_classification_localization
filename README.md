# Submission for Primaa Technical Interview - Jade Perdereau

This repository gathers my work for the two tasks given for the technical interview at Primaa.

Readme in progress
##Installation
If needed, create a custom virtual environment 

Clone the repo using git clone 

cd into current directory

Install the required packages using pip install pipreqs

## Classification of nuclei
The following assumption has been made : there are some mistakes in image labelling, notably in the test set (ie no_nuclei_120.png).
Some images were also duplicated in the training set.


## Nuclei detection

To detect nuclei, we made the following assumptions : images with no bounding boxes/ nuclei were removed from the training set.
Some labelling mistakes were also present notably in the test set (boxes equal to whole image) and they were not considered in the assessment.
