# ISRO Satellite Image Super-Resolution

This project implements a state-of-the-art super-resolution model for enhancing PROBA-V satellite imagery. The model uses an advanced transformer-based architecture with ensemble learning and quality-aware training to achieve high-quality image upscaling.

## Features

- Transformer-based super-resolution architecture
- Model ensemble with Stochastic Weight Averaging (SWA)
- Quality-aware training with mask-based loss
- Advanced data augmentation pipeline
- Mixed precision training
- Curriculum learning
- Test-time augmentation
- Comprehensive logging with Weights & Biases

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── probav_dataset.py    # Dataset loader
│   ├── models/
│   │   └── transformer_sr.py    # Model architecture
│   └── train.py                 # Training script
├── probav_data/
│   ├── train/                   # Training data
│   ├── test/                    # Test data
│   └── norm.csv                 # Normalization parameters
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
└── README.md
```

## Dataset

The project uses the PROBA-V Super-Resolution dataset from ESA's Kelvins competition. The dataset contains satellite data from 74 hand-selected regions around the globe at different points in time, consisting of:
- 300m resolution data (128x128 pixels)
- 100m resolution data (384x384 pixels)
- Quality maps for cloud/ice/water masking

You can download the dataset from [Zenodo](https://zenodo.org/records/6327426).

Place the downloaded data in the `probav_data` directory following the structure shown above.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the environment:
```bash
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the `probav_data` directory
2. Configure training parameters in `config.yaml`
3. Start training:
```bash
python src/train.py
```
## Training Features

- **Model Ensemble**: Trains multiple models in parallel and averages predictions
- **Advanced Loss**: Combines L1, MSE, and SSIM losses with quality-aware weighting
- **Data Augmentation**: Comprehensive augmentation pipeline including:
  - Random crops, flips, and rotations
  - Color jittering
  - Random erasing
  - Test-time augmentation
- **Optimization**: 
  - One Cycle learning rate
  - Mixed precision training
  - Stochastic Weight Averaging
  - Curriculum learning

## Results

The model achieves high accuracy through:
- Quality-aware training
- Ensemble predictions
- Advanced validation strategy
- Test-time augmentation

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
