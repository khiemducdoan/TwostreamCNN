# TwoStreamCNN: American Sign Language Classification

## Overview
A dual-stream CNN architecture for American Sign Language (ASL) alphabet classification using PyTorch. The project implements a two-stream convolutional neural network approach to process and classify ASL hand gestures.

## Features
- Two-stream CNN architecture
- Support for ASL alphabet classification
- TensorBoard integration for training visualization
- Data augmentation with random horizontal flips
- Configurable training parameters via YAML

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-enabled GPU (recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd TwostreamCNN

# Install dependencies
pip install -r requirements.txt
```

### Dataset
The project uses the ASL Alphabet dataset from Kaggle:
```bash
# Download dataset
kaggle datasets download -d grassknoted/asl-alphabet

# Extract dataset
unzip asl-alphabet.zip
```

## Project Structure
```
TwostreamCNN/
├── configs/             # Configuration files
│   └── config.yml      # Main configuration
├── model/              # Model implementations
│   ├── Resnet.py
│   └── TwoStreamCNN.py
├── dataset/           # Dataset handling
├── runners/           # Training runners
├── utils/             # Utility functions
└── ckpt/             # Model checkpoints
```

## Usage

### Training
```bash
python main.py
```

### Configuration
Modify `configs/config.yml` to adjust training parameters and model architecture.

## Model
The architecture consists of:
- Two parallel CNN streams
- ResNet-based feature extraction
- Custom fusion layers

## Monitoring
Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir=logs
```

## License
[Insert License Information]

## Acknowledgments
- ASL Alphabet Dataset from Kaggle