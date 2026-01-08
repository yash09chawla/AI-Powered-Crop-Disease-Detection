# 🌱 AI-Powered Crop Disease Detection

A deep learning model that predicts crop diseases from plant leaf images using Convolutional Neural Networks (CNN). This project helps farmers and agricultural experts identify plant diseases early, enabling timely intervention and better crop management.

**Kaggle Dataset Link:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 📋 Overview

This project implements a CNN-based image classification system that can identify 38 different crop disease classes across multiple plant species including Apple, Blueberry, Cherry, Corn, Grape, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

## ✨ Features

- **38 Disease Classes**: Classifies various diseases and healthy states across 13 plant species
- **High Accuracy**: Achieves 88.28% validation accuracy
- **Image Preprocessing**: Automated image preprocessing and augmentation
- **Easy Prediction**: Simple function-based interface for making predictions on new images
- **Model Persistence**: Save and load trained models for future use

## 🏗️ Model Architecture

The model uses a sequential CNN architecture:

- **Input Layer**: 224x224x3 RGB images
- **Convolutional Layers**: 
  - Conv2D (32 filters, 3x3 kernel) + MaxPooling2D
  - Conv2D (64 filters, 3x3 kernel) + MaxPooling2D
- **Dense Layers**: 
  - Flatten layer
  - Dense layer (256 neurons, ReLU activation)
  - Output layer (38 neurons, Softmax activation)

**Total Parameters**: ~47.8 million

## 📊 Results

- **Training Accuracy**: 97.60%
- **Validation Accuracy**: 88.28%
- **Training Loss**: 0.0761
- **Validation Loss**: 0.5091
- **Epochs**: 5

## 🚀 Usage

### Prerequisites

```bash
pip install tensorflow keras numpy matplotlib pillow kaggle
```

### Training the Model

1. Download the PlantVillage dataset from Kaggle
2. Set up Kaggle API credentials (`kaggle.json`)
3. Run the notebook cells to:
   - Download and extract the dataset
   - Preprocess images (resize to 224x224, normalize)
   - Split data (80% training, 20% validation)
   - Train the CNN model
   - Evaluate performance

### Making Predictions

```python
from tensorflow.keras.models import load_model
import json

# Load the trained model
model = load_model('plant_disease_prediction_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Predict on a new image
predicted_class = predict_image_class(model, 'path/to/image.jpg', class_indices)
print(f"Predicted Disease: {predicted_class}")
```

## 📁 Project Structure

```
Crop-Disease/
├── Model.ipynb              # Main training notebook
├── test_images/             # Sample test images
│   ├── test_apple_black_rot.JPG
│   ├── test_blueberry_healthy.jpg
│   └── test_potato_early_blight.jpg
└── README.md
```

## 🔧 Technical Details

- **Framework**: TensorFlow/Keras
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Image rescaling (1/255 normalization)
- **Train/Validation Split**: 80/20

## 📚 Dataset Information

The PlantVillage dataset contains:
- **Total Images**: 54,305 (43,456 training + 10,849 validation)
- **Classes**: 38 different disease/health states
- **Image Format**: Color images (256x256, resized to 224x224)
- **Plant Species**: 13 different crop types

## 🎯 Supported Plant Diseases

The model can identify diseases in:
- Apple (Black rot, Cedar apple rust, Healthy, Scab)
- Blueberry (Healthy)
- Cherry (Healthy, Powdery mildew)
- Corn (Cercospora leaf spot, Common rust, Healthy, Northern Leaf Blight)
- Grape (Black rot, Esca, Healthy, Leaf blight)
- Peach (Bacterial spot, Healthy)
- Pepper (Bacterial spot, Healthy)
- Potato (Early blight, Healthy, Late blight)
- Raspberry (Healthy)
- Soybean (Healthy)
- Squash (Powdery mildew)
- Strawberry (Healthy, Leaf scorch)
- Tomato (Bacterial spot, Early blight, Healthy, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Mosaic virus, Yellow Leaf Curl Virus)

## 📖 References

- **Dataset**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Note**: This model is trained for educational purposes. For production use in agriculture, consider additional validation and consultation with agricultural experts.
