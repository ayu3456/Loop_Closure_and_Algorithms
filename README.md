# Loop Closure Detection with Random Forest and CNN

This repository implements loop closure detection using a combination of traditional computer vision techniques (SIFT features) and deep learning (CNN features), enhanced with Random Forest classification.

## Features

- **Hybrid Feature Extraction**:
  - SIFT features for local feature matching
  - CNN features (ResNet18) for global image similarity
  - Random Forest classification for robust loop closure detection

- **Key Components**:
  - `random_forest_loop_closure.py`: Base implementation using SIFT and Random Forest
  - `cnn_loop_closure.py`: Enhanced implementation adding CNN features
  - Visualization of detected loop closures

## Requirements

```bash
numpy>=1.21.0
opencv-python>=4.5.3
scikit-learn>=0.24.2
scipy>=1.7.0
matplotlib>=3.4.2
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.3.1
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayu3456/Loop_Closure_and_Algorithms.git
cd Loop_Closure_and_Algorithms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the basic Random Forest implementation:
```bash
python random_forest_loop_closure.py
```

2. Run the CNN-enhanced version:
```bash
python cnn_loop_closure.py
```

## Algorithm Details

### Random Forest Implementation
- Uses SIFT features for keypoint detection and description
- Matches features using FLANN-based matcher
- Applies geometric verification with RANSAC
- Uses Random Forest for classification

### CNN Enhancement
- Uses ResNet18 pre-trained on ImageNet
- Extracts global image features
- Combines CNN similarity with Random Forest predictions
- Improved confidence scoring system

## Results

The system detects loop closures with:
- High confidence scores (0.507-0.765)
- Strong inlier ratios (0.305-0.425)
- Robust performance across different viewpoints

## Output

- Visualizations saved in `output/` directory
- Detailed confidence scores and inlier ratios for each detection
- Frame pair matching visualization 