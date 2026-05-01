# Cancer Detection using MobileNetV2 (Deep Learning)

## Overview
This project builds a deep learning model to classify medical images as **Cancer** or **Not Cancer** using the pre-trained MobileNetV2 architecture. It uses transfer learning and image preprocessing techniques to improve classification performance.

## Objective
The main goals of this project are:

- Automatically detect cancer from medical images.
- Assist early diagnosis using AI-based image classification.
- Demonstrate transfer learning with a lightweight convolutional neural network.
- Improve model performance through preprocessing, augmentation, and fine-tuning.

## Dataset
A custom image dataset is used for training, validation, and testing.

### Dataset Structure
```text
Dataset/
├── train/
├── val/
└── test/
```

### Classes
- `Cancer` — Images containing cancer.
- `Not_Cancer` — Images without cancer.

### Image Size
All images are resized to `224 x 224` pixels before being passed to the model.

## Technologies Used
- Python
- TensorFlow / Keras
- Google Colab / Jupyter Notebook

### Libraries
- tensorflow
- numpy
- matplotlib
- seaborn
- sklearn
- PIL

## Project Workflow

### 1. Data Loading
- Images are loaded using `ImageDataGenerator`.
- Separate directories are used for training, validation, and testing.

### 2. Data Preprocessing
- Resize images to `224 x 224`.
- Apply MobileNetV2 preprocessing.
- Use data augmentation techniques such as:
  - Rotation
  - Zoom
  - Horizontal flip

### 3. Handling Imbalance
- Class weights are computed to reduce the effect of dataset imbalance.

### 4. Model Building
- Base model: `MobileNetV2` pre-trained on ImageNet.
- Custom top layers added:
  - Global Average Pooling
  - Dense layer with ReLU activation
  - Batch Normalization
  - Dropout (`0.4`)
  - Output layer with Softmax activation

### 5. Training Strategy
Training is done in two stages.

#### Stage 1
- Freeze the base MobileNetV2 model.
- Train only the newly added classification layers.

#### Stage 2
- Unfreeze the last 30 layers of the base model.
- Fine-tune with a low learning rate.

### 6. Optimization Techniques
- `EarlyStopping` to prevent overfitting.
- `ReduceLROnPlateau` to reduce learning rate when validation performance stalls.
- `ModelCheckpoint` to save the best model.

### 7. Evaluation
Model performance is evaluated using:
- Confusion Matrix
- Classification Report
- Accuracy score

### 8. Model Saving
The final trained model is saved as:

```text
cancer_model.h5
```

## Prediction Script
The prediction script loads the trained model and predicts the class of a single medical image.

### Prediction Steps
- Load the image.
- Resize it to `224 x 224`.
- Preprocess the input using MobileNetV2 preprocessing.
- Predict the class and confidence score.

### Example Output
```text
Prediction: Cancer
Confidence: 97.45%
```

## Model Used
### MobileNetV2 (Transfer Learning)
- Lightweight and efficient CNN.
- Pre-trained on ImageNet.
- Well-suited for image classification tasks.

## Example Predictions
| Image Type | Result |
|-----------|--------|
| Cancer Image | Cancer |
| Normal Image | Not_Cancer |

## Results
- High accuracy can be achieved depending on dataset quality.
- Class imbalance is handled using computed class weights.
- Fine-tuning improves the model's performance on domain-specific images.

## Limitations
- Requires high-quality labeled medical images.
- Performance depends on dataset size and diversity.
- May not generalize well to unseen medical data.
- Should not be considered a replacement for clinical diagnosis.

## Important Disclaimer
This project is for **educational purposes only**. It must not be used for real medical diagnosis without professional validation and testing.

## Conclusion
This project demonstrates how transfer learning with MobileNetV2 can classify medical images into cancer and non-cancer categories. With proper data preparation, augmentation, and fine-tuning, deep learning models can support healthcare-related image classification tasks.