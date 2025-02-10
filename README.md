# EchoDFKD

## Overview

EchoDFKD is a project that uses deep learning techniques to segment the left ventricle in echocardiography videos in a data-free knowledge distillation setup. 

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- PyTorch 1.10+
- OpenCV
- PyTorch Lightning
- NumPy
- Pandas

You can install the required packages using the provided `requirements.txt` file:

```bash
python3 -m venv ~/echodfk
source ~/echodfk/bin/activate
pip install -r requirements.txt
```

## Directory Structure

The repository is structured as follows:

```
EchoDFKD/
│
├── a4c-video-dir/           # Directory containing video files and related data
│   ├── FileList.csv
│   ├── synthetic_FileList.csv
│   ├── Videos/
│   ├── Videos_synthetic/
│   └── VolumeTracings.csv
│
├── ConvLSTM_Segmentation/   # Subrepo containing the student model architecture
│   └── ...
│
├── core/                    # Whole pipeline
│   ├── utils.py
│   ├── metrics.py
│   └── train.py
│   └── inference.py
│   └── evaluate_LVEF.py
│   └── evaluate_DICE.py
│   └── evaluate_aFD.py
│
├── data/                    # Will store large intermediate files
│   └── ...
│
├── echoclip/                # Echoclip related data/feature files
│   └── ...
│
├── echonet_a4c_example.py   # define the important class Example
│
├── echonet_deeplab_dir/
│   └── size.csv             # ED&ES labelled frames no. for each video
│
├── examples_and_vizualisation/ 
│   ├── Study_labels.ipynb   # Visualize labels produced by humans
│   └── Study_EchoCLIP_outputs.ipynb # Visualize EchoCLIP-based phase inference
│
├── hyperparameters/         # Hyperparameter configurations
│   └── ...
│
├── models/                  # Directory for storing model weights and hyperparams
│   └── ...
│
├── Output/                  # Directory for output files
│   └── ...
│
├── README.md                # Project documentation
│
└── settings.py              # Constants, settings and configurations
```

## Prepare Your Dataset

WIP

## Configure Paths and Hyperparameters

WIP

## Run the pipeline

WIP

