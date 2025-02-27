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

0. **Production of a synthetic dataset**
1. **Production of targets on synthetic dataset**
2. **Training of the student model**
3. **Inference**
4. **Model evaluation**
5. **Visuals**

## Directory Structure

The repository is structured as follows:

```
EchoDFKD/
│
├── a4c-video-dir/             # Directory containing video files and related data
│   ├── FileList.csv           # contains volumes & EF, and train/val/test split for real data
│   ├── synthetic_FileList.csv # contains volumes & EF, and train/val/test split for synthetic data
│   ├── Videos/                # Dir containing real clips in AVI format (converted from DICOM)
│   ├── Videos_synthetic/      # Dir containing synthetic AVI videos
│   └── VolumeTracings.csv     # File from EchoNet-Dynamic containing human labels
│
├── ConvLSTM_Segmentation/     # Subrepo containing the student model architecture
│   └── ...
│
├── core/                               # Whole pipeline
│   ├── produce_targets.py              # Produces targets for synthetic dataset (first step)
│   └── train.py                        # Trains the student model (second step)
│   └── inference.py                    # Performs inference on the test dataset (third step)
│   └── evaluate_LVEF.py                # Evaluates the student model on the test set (fourth step, part 1)
│   └── evaluate_DICE.py                # Evaluates the student model on the test set (fourth step, part 2)
│   └── evaluate_aFD.py                 # Evaluates the student model on the test set (fourth step, part 3)
│   └── create_visuals.py               # Creates visuals for the student model (fifth step)
│   └.. (create_synthetic_dataset.py ?) # WIP, would be step 0
│
├── data/                               # Will store large intermediate files
│   └── ...
│
├── echoclip/                           # Echoclip related data/feature files
│   └── ...
│
├── echonet_a4c_example.py              # define the important class Example, representing a clip
│
├── echonet_deeplab_dir/
│   └── size.csv                        # ED&ES labelled frames no. for each video
│
├── examples_and_vizualisation/ 
│   ├── Study_labels.ipynb              # Visualize labels produced by humans
│   └── Study_EchoCLIP_outputs.ipynb    # Visualize EchoCLIP-based phase inference
│
├── hyperparameters/                    # Hyperparameter configurations
│   └── ...
│
├── models/                             # Directory for storing model weights and hyperparams
│   └── ...
│
├── Output/                             # Directory for model outputs
│   └── ...
│
└── settings.py                         # Constants, paths, settings
```
