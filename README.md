# Multi-Scale Spatiotemporal Attention Network (MS-STAN) for Action Recognition

## Overview

This repository contains the implementation of the **Multi-Scale Spatiotemporal Attention Network (MS-STAN)** for action recognition in **welding and joining processes of dissimilar materials**. The model leverages advanced **spatiotemporal feature extraction and attention mechanisms** to improve real-time monitoring and analysis of industrial processes.

## Features

- **Multi-Scale Feature Extraction**: Captures both fine-grained and large-scale motion patterns.
- **Spatiotemporal Attention Mechanism**: Enhances relevant features while suppressing noise.
- **Lightweight & Efficient Design**: Optimized for real-time inference in industrial applications.
- **Modular Implementation**: Well-structured code for easy adaptation and experimentation.

## Repository Structure

/MS-STAN-Action-Recognition
│── README.md               # Project introduction
│── requirements.txt        # Dependencies list
│
│── data/                   # Dataset storage
│   ├── raw/                # Raw data (videos, images)
│   ├── processed/          # Preprocessed data (frames, features)
│
│── models/                 # Trained model weights
│   └── ms_stan.pth         # Pre-trained model
│
│── src/                    # Core code
│   ├── __init__.py         # Package initialization
│   ├── dataset.py          # Data processing
│   ├── model.py            # MS-STAN model definition
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
│
│── LICENSE                 # License file

## Installation

1. **Clone the repository**  
 
   git clone https://github.com/yourusername/MS-STAN-Action-Recognition.git
   cd MS-STAN-Action-Recognition
   
  2. **Install dependencies**   
pip install -r requirements.txt


## Dataset Preparation
Place raw videos or images inside the data/raw/ directory.
Run the preprocessing script to extract frames and resize image：
python src/preprocess.py

## Training
To train the MS-STAN model, run:
python src/train.py

## Evaluation
To evaluate the trained model:
python src/evaluate.py

## Inference
For real-time predictions on new data, add an inference script (inference.py) and run:

python src/inference.py --input path_to_video_or_image



## Contributions
Code Implementation: Provided modular code structure for easy customization and reuse.

Model Enhancement: Integrated multi-scale spatiotemporal attention to improve action recognition performance.

Industrial Adaptation: Tailored for welding and joining scenarios with real-time inference capabilities.


## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgments
This project is based on deep learning methods for industrial action recognition and integrates techniques from computer vision, sequence modeling, and attention-based networks. 

