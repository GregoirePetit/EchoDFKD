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

You can install all these packages using the provided `requirements.txt` file:

```bash
python3 -m venv ~/echodfk
source ~/echodfk/bin/activate
pip install -r requirements.txt
```

## Prepare models

You will need to download some teachers model weights.
In our experiment we use the trained model from the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) project, which is available on the following link : https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt .
Don't hesitate to try other teacher models.
Place weight files in models/your_teacher_name (for instance, models/echonet_deeplabV3).

## Prepare Your Datasets

You can download a synthetic dataset on https://huggingface.co/HReynaud (or you might want to generate your own synthetic dataset).
If you want to run the experiments that show the performance of the model on the EchoNet-Dynamic dataset, you also need to download the dataset from the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) website. The dataset is available for free but you need to request access. Recently, that dataset was also available on a Kaggle link.


## Configure Paths and Hyperparameters

WIP

## Run the pipeline

The pipeline follows these steps:

0. **Production of a synthetic dataset**:
1. **Production of targets on synthetic dataset**:

2. **Training of the student model**:

3. **Inference**:
   Uses the trained student model to perform inference on the test dataset.

4. **Model evaluation**:
    Evaluates the student model on the test dataset using the following metrics:
    - **Dice Similarity Coefficient (DICE)**: A measure of the overlap between the predicted and ground truth segmentations.
    - **Average Hausdorff Distance (aFD)**: A measure of the distance between the predicted and ground truth segmentations.
    - **Left Ventricular Ejection Fraction (LVEF)**: A measure of the heart's pumping capacity.
5. **Visuals**:

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
