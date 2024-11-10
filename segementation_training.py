import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.filters import scharr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pickle

# Define ranges for Gabor filter parameters
freq_range = np.arange(0.2, 0.6, 0.05)
orient_range = [np.pi/4, 3*np.pi/4]
sigma_range = [3]
gamma_range = np.arange(0.2, 0.9, 0.05)

# Function to extract Gabor and edge features from an image
def feature_extraction(img):
    df = pd.DataFrame()
    df['Original Pixels'] = img.reshape(-1)  # Flattened original pixel values
    num = 1
    kernels = []
    
    # Create Gabor filters with varying parameters
    for theta in orient_range:
        for sigma in sigma_range:
            for lamda in freq_range:
                for gamma in gamma_range:
                    gabor_label = f'Gabor{num}'
                    kernel = cv2.getGaborKernel((31, 31), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel).reshape(-1)
                    df[gabor_label] = filtered_img  # Store filtered values in the dataframe
                    num += 1
    
    # Compute Scharr edge feature
    edge_scharr = scharr(img).reshape(-1)
    df['Scharr'] = edge_scharr

    return df

# Function to create a mask for green, yellow, and brown areas
def GYB_mask(cv2_img):
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    lower_yellow = (30, 255, 255)
    upper_yellow = (30, 255, 255)
    lower_brown = (10, 30, 30)
    upper_brown = (20, 255, 255)

    # Create masks for each color
    mask_green = cv2.inRange(cv2_img, lower_green, upper_green)
    mask_yellow_brown = cv2.inRange(cv2_img, lower_yellow, upper_yellow) + cv2.inRange(cv2_img, lower_brown, upper_brown)

    # Combine masks and refine with morphological closing
    mask = cv2.bitwise_or(mask_green, mask_yellow_brown)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to image and return masked output
    return cv2.bitwise_and(cv2_img, cv2_img, mask=mask)[:,:,2]

# Load and preprocess training images for feature extraction
path = "data1/train/"
train_dataset = pd.DataFrame()

for train_img in os.listdir(path):
    if train_img != '.DS_Store':
        input_img = cv2.imread(path + train_img)
        input_img_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        masked_img = GYB_mask(input_img_hsv)
        features_df = feature_extraction(masked_img)
        features_df['image_name'] = train_img
        train_dataset = train_dataset.append(features_df)

# Load and preprocess masks for labeling
mask_df = pd.DataFrame()
path_to_masks = 'data1/mask/'

for mask_img in os.listdir(path_to_masks):
    if mask_img != '.DS_Store':
        input_img = cv2.imread(path_to_masks + mask_img)
        grayscale_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        if "gsv" not in mask_img:
            grayscale_img[grayscale_img < 5] = 0
            grayscale_img[grayscale_img > 0] = 1
        
        mask_data_frame = pd.DataFrame()
        mask_data_frame['Label_Value'] = grayscale_img.reshape(-1)
        mask_data_frame['mask_name'] = mask_img
        mask_df = mask_df.append(mask_data_frame)

# Combine features and labels into a dataset
dataset = pd.concat([train_dataset, mask_df], axis=1)
dataset.loc[dataset['Original Pixels'] == 0, 'Label_Value'] = 0  # Remove labels where original pixels are 0

# Prepare feature and label matrices
X = dataset.drop(labels=['image_name', 'mask_name', 'Label_Value'], axis=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
Y = dataset["Label_Value"].values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate model accuracy
prediction_test = model.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, prediction_test))

# Display feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:\n", feature_importances)

# Save the trained model
filename = 'greenery_segmentation'
pickle.dump(model, open(filename, 'wb'))











