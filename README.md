# Skin-Cancer-Detection-using-TensorFlow
This code implements an image classification pipeline for identifying cancerous tissue using a deep learning approach. It utilizes the EfficientNetB7 model, a powerful pre-trained convolutional neural network, as a base and adds custom layers to tailor it for this specific task.

## Workflow:

### Data Loading and Preprocessing:

Images of cancerous (malignant) and non-cancerous (benign) tissue are loaded from specified folders.
The image paths and labels are organized into a Pandas DataFrame.
A binary label column is created to represent the two classes (malignant = 1, benign = 0).

### Data Visualization:

A pie chart is generated to visualize the distribution of malignant and benign samples in the dataset.
Sample images from each category are displayed to provide a visual understanding of the data.

### Data Splitting:

The dataset is split into training and validation sets using train_test_split from scikit-learn.
This ensures that the model is evaluated on unseen data to assess its generalization ability.

### Data Input Pipeline:

A function decode_image is defined to decode and preprocess images before feeding them to the model.
This includes resizing the images to a fixed size (224x224) and normalizing pixel values.
TensorFlow Datasets are created for efficient data loading and batching during training and validation.

### Model Definition:

The create_model function defines the architecture of the CNN model.
It loads the pre-trained EfficientNetB7 model and freezes its layers to prevent them from being updated during initial training.
Custom layers, including flattening, dense layers, batch normalization, and dropout, are added on top of the pre-trained model to enable classification.
The final model is created by specifying the inputs and outputs.

### Model Compilation and Training:

The model is compiled with the BinaryCrossentropy loss function, Adam optimizer, and AUC (Area Under the Curve) metric.
The model is trained using the training dataset and validated using the validation dataset for a specified number of epochs.

### Model Evaluation:

The training history is stored in a DataFrame and used to plot the loss and AUC values during training and validation.
These plots help visualize the model's performance and identify potential issues like overfitting.

## Overall Goal:

The code aims to develop a robust image classification model that can accurately distinguish between cancerous and non-cancerous tissue. By leveraging the power of deep learning and a pre-trained model like EfficientNetB7, it seeks to automate the process of cancer detection and assist in medical diagnosis.

## Dataset:
The dataset used can be found in this link: https://drive.google.com/drive/folders/1z45H00A-YtmtjxaWWrCtD9tzB8UZ0DaW?usp=drive_link
