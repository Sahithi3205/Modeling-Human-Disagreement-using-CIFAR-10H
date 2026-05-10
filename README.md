# Predicting Human Annotator Disagreement

## Project Overview

This project focuses on predicting human disagreement in image classification using the CIFAR-10H dataset. Unlike traditional image classification models that predict only one correct label, this project predicts probability distributions (soft labels) to model human uncertainty and disagreement.

The project was implemented using PyTorch in Google Colab.


# Datasets Used:

## CIFAR-10
CIFAR-10 is a standard image classification dataset containing:

* 60,000 RGB images
* 10 image classes
* Image size: 32×32

### Classes Include

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck


## CIFAR-10H

CIFAR-10H is an extension of CIFAR-10 that contains human annotation distributions instead of single labels.

### Example Soft Label
[0.6, 0.3, 0.1, 0, 0, ...]


This means:

* 60% of annotators selected one class
* 30% selected another class
* 10% selected a different class

Official Dataset Repository:

[CIFAR-10H Repository]
https://github.com/jcpeterson/cifar-10h



# Hard Labels vs Soft Labels

## Hard Labels

Hard labels contain only one correct class.

Example:
[0,0,0,1,0,0,0,0,0,0]


Characteristics:

* Only one class is correct
* No uncertainty
* Used in traditional classification



## Soft Labels

Soft labels contain probability distributions across multiple classes.

Example:
[0.6,0.3,0.1,0,0,...]

Characteristics:

* Multiple classes can have probability
* Captures human uncertainty
* Models disagreement between annotators



# Project Pipeline


CIFAR-10 Images + CIFAR-10H Soft Labels
                    ↓
             Data Preprocessing
                    ↓
        Train / Validation / Test Split
                    ↓
                 DataLoader
                    ↓
               ResNet-18 Model
                    ↓
               Log Softmax
                    ↓
            KL Divergence Loss
                    ↓
               Model Training
                    ↓
          Validation and Logging
                    ↓
              Final Evaluation



# Model Used

The project uses ResNet-18 as the base CNN architecture.

## Why ResNet-18?

* Strong image feature extraction
* Residual connections solve vanishing gradient problem
* Good performance for image classification



# Architecture Modifications

The original ResNet-18 architecture was modified for CIFAR-10 images.


## Modifications Made

* Final fully connected layer changed to output 10 classes
* Dropout layer added to reduce overfitting
* Small-image optimization for CIFAR-10


# What is Softmax?

Softmax converts model outputs (logits) into probabilities.

Example:

Before Softmax:
[2.1, 0.5, -1.2]

After Softmax:
[0.75, 0.20, 0.05]

Properties:

* Values between 0 and 1
* Total probability sum = 1



# What is LogSoftmax?

LogSoftmax is the logarithm of softmax probabilities.

Used because:
KLDivLoss expects log probabilities

Code used:
preds = F.log_softmax(logits, dim=1


# Loss Function

KL Divergence Loss was used for training.
loss_fn = nn.KLDivLoss(reduction='batchmean')

## Why KL Divergence?

KL Divergence compares probability distributions.

This project predicts:

* Human probability distributions
* Soft labels

Jensen-Shannon Divergence (JSD):
Jensen-Shannon Divergence was also used to compare predicted and target probability distributions.

Why JSD?
JSD is a symmetric and more stable version of KL Divergence.

It helps:
measure similarity between distributions
improve stability during distribution comparison
better analyze human disagreement patterns

JSD is useful because this project focuses on uncertainty and soft-label learning.

Custom Entropy-Aware Loss:
A custom entropy-aware loss function was used to handle uncertain and ambiguous samples more effectively.

Why Entropy-Aware Loss?

Some images have:
low disagreement (clear images)
high disagreement (ambiguous images)

The entropy-aware loss gives additional importance to:
difficult samples
highly uncertain images
images with greater annotator disagreement

This helps the model better learn human uncertainty and ambiguity.

# Training Details

| Parameter     | Value |
| ------------- | ----- |
| Optimizer     | Adam  |
| Learning Rate | 1e-4  |
| Epochs        | 20    |
| Batch Size    | 64    |
| Weight Decay  | 1e-4  |


# Training Process

The model training process included:

1. Forward propagation
2. LogSoftmax application
3. KL divergence loss calculation
4. Backpropagation
5. Weight updates using Adam optimizer



# Validation and Logging

During training, the following were monitored:

* Training Loss
* Validation Loss
* Test Loss

Example logging output:
Epoch 1/20 | Train Loss: 0.04 | Val Loss: 1.20

Purpose of logging:

* Monitor learning
* Detect overfitting
* Compare train and validation performance



# Evaluation

Final evaluation was performed using:
* Test Loss
* KL Divergence
* Jensen-Shannon Divergence (JSD)
* Custom Entropy-Aware Loss
* Probability distribution comparison

Example:
Test Loss ≈ 1.27

Lower loss indicates better prediction quality.


# Sample Output

Example model prediction:
Prediction: [0.0047, 0.1188, ..., 0.7910, ...]
Target:     [0, 0, 0, 1, 0, ...]


The model predicts probability distributions instead of a single class label.


# Results

## Observations

* Training loss decreased steadily
* Validation loss remained higher than training loss
* Slight overfitting was observed
* The model successfully learned soft-label distributions


# Repository Structure

Predcting-Human-Annotator-Disagreement/
│
├── data/              # CIFAR-10H dataset files
├── models/            # Saved trained models (.pth)
├── notebooks/         # Google Colab / Jupyter notebooks
├── outputs/           # Graphs, plots, evaluation metrics
├── README.md          # Project overview and instructions
├── requirements.txt   # Required Python libraries
└── report/            # Final project report


# How to Run

## 1. Clone Repository
git clone https://github.com/Sahithi3205/Predcting-Human-Annotator-Disagreement.git


## 2. Install Required Libraries
pip install torch torchvision numpy matplotlib scipy


## 3. Open Google Colab or Jupyter Notebook
jupyter notebook

## 4. Run Notebooks in Order

* Data loading & preprocessing
* Baseline model training
* Improved model training
* Evaluation metrics
* Visualization and outputs


# Repository Link

[Project Repository]
https://github.com/Sahithi3205/Predcting-Human-Annotator-Disagreement


# Concepts Used

* Soft Labels
* Hard Labels
* Human Uncertainty Modeling
* Entropy
* KL Divergence
* Deep Learning
* CNNs
* ResNet-18
* PyTorch
* LogSoftmax
* Model Evaluation


# Conclusion

This project demonstrates how deep learning models can be trained to capture human disagreement and uncertainty in image classification tasks.
Instead of predicting only one correct class, the model learns probability distributions using CIFAR-10H soft labels and KL Divergence loss.
The project successfully models human uncertainty using deep learning and modified ResNet-18 architecture.


# References

1. [CIFAR-10H Repository]
   https://github.com/jcpeterson/cifar-10h
2. [CIFAR-10 Dataset Information]
   https://www.cs.toronto.edu/~kriz/cifar.html
3. [Project GitHub Repository]
   https://github.com/Sahithi3205/Predcting-Human-Annotator-Disagreement
