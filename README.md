# Face Emotion Detection using CNN

A deep learning project for facial emotion classification using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Overview

This project implements a CNN-based image classification pipeline for recognizing facial expressions across seven emotion classes.

The notebook covers:

- Dataset extraction and organization
- Image preprocessing
- Grayscale image normalization
- CNN architecture design
- Model training with Adam optimization
- Emotion prediction
- Classification evaluation
- Confusion-matrix analysis

## Dataset

The notebook uses a FER dataset organized into training and testing directories.

Dataset statistics used in the project:

- Training images: 28,709
- Test images: 7,178
- Emotion classes: 7
- Image size: 64 × 64 pixels
- Image mode: Grayscale

The dataset itself is **not included** in this repository.

## CNN Architecture

The implemented CNN contains three convolutional blocks:

1. Conv2D — 32 filters
2. Batch Normalization
3. Max Pooling
4. Dropout — 0.25

5. Conv2D — 64 filters
6. Batch Normalization
7. Max Pooling
8. Dropout — 0.25

9. Conv2D — 128 filters
10. Batch Normalization
11. Max Pooling
12. Dropout — 0.30

The classification head contains:

- Flatten
- Dense — 256 units
- Dropout — 0.50
- Dense — 7 units with Softmax

The model contains approximately 1.27 million parameters.

## Preprocessing

The images used for CNN training are:

- Resized to 64 × 64
- Converted/handled as grayscale images
- Rescaled by `1./255`

## Training Configuration

The CNN was trained using:

- Optimizer: Adam
- Learning rate: 0.0003
- Loss: Categorical Cross-Entropy
- Epochs: 25
- Metric: Accuracy

## Evaluation Results

The recorded CNN evaluation results are:

| Metric | Result |
|---|---:|
| Training Accuracy | 82.34% |
| Test Accuracy | 59.71% |
| Train-Test Gap | 22.63% |

The 22.63 percentage-point gap indicates that the model may be overfitting.

### Class-wise CNN Accuracy

| Emotion | Accuracy | Correct / Total |
|---|---:|---:|
| Angry | 44.57% | 427 / 958 |
| Disgust | 42.34% | 47 / 111 |
| Fear | 34.67% | 355 / 1024 |
| Happy | 84.39% | 1497 / 1774 |
| Neutral | 60.58% | 747 / 1233 |
| Sad | 46.99% | 586 / 1247 |
| Surprise | 75.45% | 627 / 831 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Angry | 0.55 | 0.45 | 0.49 | 958 |
| Disgust | 0.81 | 0.42 | 0.56 | 111 |
| Fear | 0.45 | 0.35 | 0.39 | 1024 |
| Happy | 0.75 | 0.84 | 0.80 | 1774 |
| Neutral | 0.51 | 0.61 | 0.55 | 1233 |
| Sad | 0.47 | 0.47 | 0.47 | 1247 |
| Surprise | 0.73 | 0.75 | 0.74 | 831 |

**Overall accuracy:** 0.60 on 7,178 test images.

**Macro average:** Precision 0.61 · Recall 0.56 · F1-Score 0.57

**Weighted average:** Precision 0.59 · Recall 0.60 · F1-Score 0.59

## Repository Structure

```text
face-emotion-detection-cnn/
│
├── notebook/
│   └── Face_Emotion_Detection_CNN.ipynb
│
├── images/
│   ├── evaluation_report.png
│   ├── train_confusion_matrix.png
│   ├── validation_confusion_matrix.png
│   └── test_confusion_matrix.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

The `images/` folder contains visual summaries of the existing notebook evaluation outputs. If a particular image is not present in the notebook, it should not be added just to fill the structure.

## Installation

Create and activate a Python virtual environment if desired, then install the project dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

The main project artifact is the Jupyter/Google Colab notebook:

```text
notebook/Face_Emotion_Detection_CNN.ipynb
```

The original notebook uses dataset files and paths configured for the Colab environment. To run it locally, update the dataset paths to match the local environment.

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Limitations and Future Improvements

The evaluation shows a substantial train-test gap, indicating overfitting. Possible improvements include:

- Data augmentation
- Hyperparameter tuning
- Regularization improvements
- Transfer learning
- More systematic validation
- Improving class-wise performance
- Real-time inference/deployment

---
⭐ Support

If you like this project, give it a ⭐ on GitHub!

## Author

**Abdul Basit**

BS Artificial Intelligence — KIET
