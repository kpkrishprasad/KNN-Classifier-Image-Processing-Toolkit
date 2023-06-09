# KNN Image Classifier with Image Processing Toolkit

This repository contains Python scripts for an image classifier based on the K-Nearest Neighbors (KNN) algorithm. It also includes a set of utility functions for image processing tasks.

## Files

- `knn_image_classifier.py`: This file contains the implementation of the KNN image classifier. It loads a dataset of labeled images, trains the KNN model, and performs image classification based on user input.

- `image_processing_utils.py`: This file contains a collection of utility functions for image processing. It includes functions for resizing images, converting them to grayscale, and extracting features.

## Usage

To use the KNN image classifier and image processing toolkit, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Run the `knn_image_classifier.py` script using the following command:
   ```bash
   python knn_image_classifier.py
   ```
   This will execute the KNN image classifier. Follow the on-screen instructions to input the image you want to classify.

3. If you need to perform additional image processing tasks, you can utilize the functions provided in `image_processing_utils.py`. Import the module into your own Python script and call the necessary functions.

## Dependencies

The following dependencies are required to run the scripts:

- Python 3.x
- OpenCV (cv2)
- NumPy

You can install these dependencies by running `pip install -r requirements.txt`.

## License

This project is licensed under the [MIT License](LICENSE).
