## Greenery Segmentation with Gabor Filters and Random Forest

This repository contains code for segmenting greenery regions in images by applying Gabor filters for feature extraction and using a Random Forest classifier for pixel classification. The project is designed to classify pixels as part of greenery regions (green, yellow, or brown vegetation) or as non-greenery, aiding in vegetation segmentation tasks.

The project performs segmentation on images to identify and classify greenery regions. It uses Gabor filters to extract texture features based on frequency, orientation, and other parameters. The model then leverages a Random Forest classifier to predict greenery segmentation on test images.


## Feature Extraction

The feature_extraction function extracts features from images using:

Gabor Filters: Applies a set of Gabor filters to capture texture features based on different orientations, frequencies, and other parameters.
Scharr Filter: Detects edges in the image, enhancing feature detection along boundaries.

## Training and Evaluation

Training: Uses a 60-40 train-test split to train the Random Forest classifier on the extracted features.
Evaluation: Prints the accuracy of the model and displays feature importances in the console.

## Results

The trained model’s accuracy is displayed after running the script, providing insights into its effectiveness in segmenting greenery regions. Additionally, feature importances are printed to understand which features had the most significant impact on classification.

## Model Saving

The trained Random Forest model is saved as greenery_segmentation in the project directory. This model can be loaded for future predictions.

## Contributing

Feel free to fork this repository, make enhancements, and submit pull requests. Contributions are welcome!
