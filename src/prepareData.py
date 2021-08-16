# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 22:45:26 2021

@author: SUBHADEEP
"""
"""
Prepare dataset
1. Load data
2. Split into train, test, and val dataset (70:20:10) split
3. Peek
"""
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import random
import numpy as np

#// Mention your CSV file path here
annotationLocation = 'C:/Users/SUBHADEEP/Desktop/Ellipse Detection/Final_New/GT_Labels.csv'

#// Read the CSV file
annotationFile = pd.read_csv(annotationLocation)

#// Display the number of files present
print("records present in the full DataFrame: ", annotationFile.shape)

#// Peek at the annotation file
annotationFile.head()

#// Divide randomly into train, test, and validation data: 70:20:10 split
trainingDF = annotationFile.sample(frac = 0.70)

#// Process rest of the DF
rest30 = annotationFile.drop(trainingDF.index)

#// Create test DF
testDF = rest30.sample(frac = 0.6667)

#// Create validation DF
valDF = rest30.drop(testDF.index)

#// Peek at the training DF
print("records present in the training DataFrame: ", trainingDF.shape)
trainingDF.head()

#// Peek at the test DF
print("records present in the testing DataFrame: ", testDF.shape)
testDF.head()

#// Peek at the validation DF
print("records present in the validation DataFrame: ", valDF.shape)
valDF.head()

#// Read one random training data
trainingDFShape = np.shape(trainingDF)
randomIdx = random.randint(0, trainingDFShape[0])
fileName = trainingDF.iloc[randomIdx]['imgName']
randomTrainingSample = cv2.imread(fileName)

#// Read one random test data
testDFShape = np.shape(testDF)
randomIdx = random.randint(0, testDFShape[0])
fileName = testDF.iloc[randomIdx]['imgName']
randomTestSample = cv2.imread(fileName)

#// Read one random validation data
valDFShape = np.shape(valDF)
randomIdx = random.randint(0, valDFShape[0])
fileName = valDF.iloc[randomIdx]['imgName']
randomValSample = cv2.imread(fileName)

#// Visualize images
plt.figure
plt.subplot(1, 3, 1)
plt.title('Training image')
plt.imshow(randomTrainingSample)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title('Test image')
plt.imshow(randomTestSample)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title('Validation image')
plt.imshow(randomValSample)
plt.axis("off")
plt.show()
