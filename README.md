# NIH Chest X Ray Multi-Label Multi-Class Classification

## Background
Chest X-ray imaging is a common diagnostic tool for detecting various thoracic diseases and abnormalities. The National Institutes of Health (NIH) Chest X-ray dataset contains a large collection of labeled X-ray images, each associated with multiple findings and diseases. Leveraging deep learning techniques on this dataset can significantly aid in automated disease detection and classification, reducing the burden on radiologists and improving diagnostic accuracy.

## Problem Description
The objective of this project is to develop a deep learning model capable of accurately detecting and classifying thoracic diseases and abnormalities from NIH Chest X-ray images. The model should be able to handle multi-label classification, as each image may contain multiple disease findings. Key challenges include handling class imbalance, interpreting ambiguous findings, and ensuring robust generalization to unseen data.

## About Dataset

**Source:** [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)

### NIH Chest X-ray Dataset Overview
The NIH Chest X-ray dataset comprises 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning.

### Data Limitations
- The image labels are NLP extracted, so there could be some erroneous labels.
- Very limited numbers of disease region bounding boxes.
- Original radiology reports are not publicly available.

### File Contents
- **Image format:** 112,120 total images (1024 x 1024)
- **Zip files containing images:**
  - `images_001.zip`: Contains 4,999 images
  - `images_002.zip`: Contains 10,000 images
  - `images_003.zip`: Contains 10,000 images
  - `images_004.zip`: Contains 10,000 images
  - `images_005.zip`: Contains 10,000 images
  - `images_006.zip`: Contains 10,000 images
  - `images_007.zip`: Contains 10,000 images
  - `images_008.zip`: Contains 10,000 images
  - `images_009.zip`: Contains 10,000 images
  - `images_010.zip`: Contains 10,000 images
  - `images_011.zip`: Contains 10,000 images
  - `images_012.zip`: Contains 7,121 images
- **Additional files:**
  - `README_ChestXray.pdf`: Original README file
  - `BBox_list_2017.csv`: Bounding box coordinates.
  - `Data_entry_2017.csv`: Class labels and patient data for the entire dataset.

### Class Descriptions
There are **15 classes** (14 diseases and one "No findings"):
- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule Mass
- Hernia

## Goals and Deliverables

1. Preprocess the NIH Chest X-ray dataset (data cleaning, augmentation, normalization).
2. Design and train a deep learning model (such as a convolutional neural network) for multi-label classification.
3. Implement evaluation metrics (precision, recall, F1-score, AUC-ROC) for model performance assessment.
4. Validate the model on a separate test set and fine-tune hyperparameters.
5. Generate insights into model predictions (e.g., visualizations of attention maps).
6. Document the entire workflow including code implementation and evaluation results.
7. Demonstrate the model's utility in assisting radiologists with disease detection.

### Expected Impact
This project aims to develop an accurate deep learning model for thoracic disease detection and classification, enhancing medical diagnostic capabilities and improving patient outcomes through early disease detection.

## Libraries Used
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import os
import glob
import cv2
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
```
