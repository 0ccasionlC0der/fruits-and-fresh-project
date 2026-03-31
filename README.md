# Food Freshness Detection using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify food images as fresh or rotten.

The model is trained on an image dataset of fruits and learns visual patterns such as color, texture, and shape to determine freshness.

This project demonstrates the application of Deep Learning in real-world problems like food quality analysis and waste reduction.

---

## Features
- CNN-based image classification model
- Automatic feature extraction using convolutional layers
- GPU support (if available)
- Prediction with confidence score
- Clean and modular implementation
- Real-time image testing using local file input

---

## Dataset Structure
The dataset is organized as:

dataset/
├── train/
│   ├── freshapples/
│   ├── freshbanana/
│   ├── freshoranges/
│   ├── rottenapples/
│   ├── rottenbanana/
│   └── rottenoranges/
└── test/
    ├── freshapples/
    ├── freshbanana/
    ├── freshoranges/
    ├── rottenapples/
    ├── rottenbanana/
    └── rottenoranges/

---

## Model Architecture
The CNN model consists of:

- 3 Convolutional layers (Conv2D + ReLU + MaxPooling)
- Fully connected layer (256 neurons)
- Dropout layer for regularization
- Output layer for multi-class classification

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- PIL (Python Imaging Library)
- Matplotlib

---

## Installation

1. Clone the repository
git clone https://github.com/your-username/food-freshness-detection.git

2. Navigate to the project folder
cd food-freshness-detection

3. Create a virtual environment
python -m venv venv

4. Activate environment (Windows)
venv\Scripts\Activate

5. Install dependencies
pip install torch torchvision matplotlib

---

## Usage

Run the prediction script:
python predict.py

The model will:
- Load trained weights
- Accept an image path
- Predict freshness
- Display confidence score

---

## Example Output

Prediction: Fresh Apple 🍎  
Confidence: 84.23%

---

## Results
- Achieved ~87% accuracy on test dataset
- Successfully classifies multiple fruit categories

---

## Limitations
- Performance depends on dataset quality
- May misclassify images under poor lighting or unusual conditions

---

## Future Improvements
- Increase dataset size
- Add data augmentation
- Use transfer learning (ResNet, EfficientNet)
- Build web or mobile interface

---

## Author
Abha Singh

---

## License
This project is for academic and learning purposes.
