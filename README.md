# Predicting Human Annotator Disagreement using CIFAR-10H

## Project Overview

This project focuses on predicting human disagreement in image classification using the CIFAR-10H dataset. Unlike traditional image classification models that predict only one correct label, this project predicts probability distributions (soft labels) to model human uncertainty and disagreement.

The project was implemented using PyTorch in Google Colab.

---

# Datasets Used

## CIFAR-10
CIFAR-10 is a standard image classification dataset containing:
- 60,000 RGB images
- 10 image classes
- Image size: 32×32

Classes include:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## CIFAR-10H
CIFAR-10H is an extension of CIFAR-10 that contains human annotation distributions instead of single labels.

Example soft label:

```python
[0.6, 0.3, 0.1, 0, 0, ...]
```

This means:
- 60% of annotators selected one class
- 30% selected another class
- 10% selected a different class

Official Dataset Repository:  
https://github.com/jcpeterson/cifar-10h

---

# Project Pipeline

```text
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
```

---

# Model Used

The project uses ResNet-18 as the base CNN architecture.

Modifications made:
- Final fully connected layer changed to output 10 classes
- Dropout layer added to reduce overfitting

```python
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 10)
)
```

---

# Loss Function

KL Divergence Loss was used for training.

```python
loss_fn = nn.KLDivLoss(reduction='batchmean')
```

KL Divergence was chosen because CIFAR-10H provides probability distributions instead of hard labels.

---

# Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 20 |
| Batch Size | 64 |
| Weight Decay | 1e-4 |

---

# Logging and Evaluation

During training, the following were monitored:
- Training Loss
- Validation Loss
- Test Loss

Loss graphs were plotted to analyze model learning and overfitting.

---

# Sample Output

Example model prediction:

```python
Prediction: [0.0047, 0.1188, ..., 0.7910, ...]
Target:     [0, 0, 0, 1, 0, ...]
```

The model predicts probability distributions instead of a single class label.

---

# Results

### Observations
- Training loss decreased steadily
- Validation loss remained higher than training loss
- Slight overfitting was observed
- The model successfully learned soft-label distributions

Example:

```python
Test Loss ≈ 1.27
```

---

# Repository Structure

```text
Predcting-Human-Annotator-Disagreement/
│
├── notebooks/
├── outputs/
├── models/
├── README.md
├── requirements.txt
└── report/
```

---

# How to Run

## Clone Repository

```bash
git clone https://github.com/Sahithi3205/Predcting-Human-Annotator-Disagreement.git
```

## Install Required Libraries

```bash
pip install torch torchvision numpy matplotlib
```

## Run Notebook

Open the notebook in Google Colab and run all cells sequentially.

---

# Concepts Used

- Soft Labels
- Human Uncertainty Modeling
- KL Divergence
- Deep Learning
- CNNs
- ResNet-18
- PyTorch

---

# Conclusion

This project demonstrates how deep learning models can be trained to capture human disagreement and uncertainty in image classification tasks. Instead of predicting only one correct class, the model learns probability distributions using CIFAR-10H soft labels and KL Divergence loss.

---

# References

1. https://github.com/jcpeterson/cifar-10h  
2. https://www.cs.toronto.edu/~kriz/cifar.html  
3. https://github.com/Sahithi3205/Predcting-Human-Annotator-Disagreement
