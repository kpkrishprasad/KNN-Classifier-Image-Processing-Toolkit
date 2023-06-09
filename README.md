# KNN Image Classifier with Image Processing Toolkit

This repository contains Python scripts for an image classifier based on the K-Nearest Neighbors (KNN) algorithm. It also includes a set of utility functions for image processing tasks.

## Files

- image_processing_utils.py: This file contains a comprehensive set of utility functions for various image processing tasks. It provides a range of functionalities to preprocess and manipulate images before feeding them into the KNN image classifier. Some of the key functions include:

Resizing images: It includes functions to resize images to a specific width and height or scale them proportionally while maintaining the aspect ratio.

Converting to grayscale: It provides functions to convert color images to grayscale, which can be useful for certain image analysis tasks or reducing computational complexity.

Extracting features: The file includes functions to extract features from images, such as color histograms, texture descriptors, or shape-based features. These features can be used as input for the KNN classifier or other machine learning algorithms.

Applying filters: It offers a collection of functions to apply various filters to images, such as blurring, sharpening, edge detection, or noise reduction. These filters can enhance image quality or highlight specific features.


- image_processing_utils.py: This file contains a comprehensive set of utility functions for various image preprocessing and manipulation tasks. It provides a wide range of functionalities to preprocess and enhance images before using them for classification or other analysis purposes. Some of the key tasks supported by the file include:

Image resizing: The file includes functions to resize images to specific dimensions or scale them proportionally while preserving the aspect ratio. Resizing images can be helpful to standardize the input size for the KNN image classifier or to match the required dimensions of downstream image processing algorithms.

Grayscale conversion: It offers functions to convert color images to grayscale. Grayscale images contain a single channel representing the intensity values, which can be useful for certain image analysis tasks or when color information is not required. Converting images to grayscale reduces computational complexity and can improve the efficiency of subsequent processing steps.

Feature extraction: The file provides functions to extract various image features that can be used as inputs for the KNN classifier or other machine learning algorithms. These features include color histograms, texture descriptors, edge maps, or shape-based features. Extracting informative features from images can enhance the discriminative power of the classifier and improve classification accuracy.

Image filtering: It includes functions to apply different filters and transformations to images. These filters can be used for tasks such as blurring, sharpening, noise reduction, or edge detection. Filtering techniques help to enhance image quality, reduce noise, emphasize edges, or highlight specific features, thus improving the overall performance of the KNN classifier.

Image augmentation: The file provides functions to perform image augmentation techniques, such as rotation, translation, scaling, or flipping. Image augmentation is particularly useful when working with limited training data, as it generates additional variations of existing images, thereby increasing the diversity and robustness of the training set.

By leveraging the functionalities provided by the image_processing_utils.py file, you can preprocess and manipulate images effectively, tailoring them to the requirements of the KNN image classifier and other image analysis tasks within your project.
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
