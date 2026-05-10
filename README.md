Modeling Human Disagreement using CIFAR-10H

This project explores human uncertainty in image classification using the CIFAR-10H dataset. Unlike traditional image classification tasks that use hard labels, this project trains a deep learning model to predict probability distributions (soft labels) representing disagreement among human annotators.

The project uses:

CIFAR-10 → image dataset
CIFAR-10H → human soft-label distributions

The implementation is built using PyTorch and trained in Google Colab.

Project Objective

Traditional classifiers assume:

One image → One correct label

However, humans often disagree on ambiguous images.

Example:

Cat: 70%
Dog: 20%
Deer: 10%

Instead of predicting a single class, this project predicts a distribution over classes, capturing human perceptual uncertainty.

The CIFAR-10H dataset was introduced to study human uncertainty in image classification.

Dataset Information
CIFAR-10
60,000 RGB images
10 classes
Image size: 32×32
50,000 training images
10,000 test images

CIFAR-10 is one of the most widely used image classification benchmarks.

CIFAR-10H

CIFAR-10H extends CIFAR-10 by providing:

Multiple human annotations per image
Soft-label distributions
Human disagreement information

Example soft label:

[0.6, 0.3, 0.1, 0, 0, ...]

Official dataset repository:
CIFAR-10H GitHub Repository

Project Pipeline
CIFAR-10 Images + CIFAR-10H Labels
                ↓
         Data Preprocessing
                ↓
        Train / Val / Test Split
                ↓
             DataLoader
                ↓
           ResNet-18 Model
                ↓
           Log Softmax
                ↓
        KL Divergence Loss
                ↓
           Training Loop
                ↓
      Validation + Logging
                ↓
           Final Evaluation
Technologies Used
Python
PyTorch
Torchvision
NumPy
Matplotlib
Google Colab
Model Architecture

The project uses ResNet-18 as the base architecture.

Modifications:

Final fully connected layer modified for 10 classes
Dropout added for regularization
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 10)
)
Loss Function

We use:

nn.KLDivLoss(reduction='batchmean')

Why KL Divergence?

CIFAR-10H contains probability distributions
KL divergence compares distributions
Better suited for soft-label learning than CrossEntropy
Training Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	1e-4
Weight Decay	1e-4
Epochs	20
Batch Size	64
Logging and Evaluation

The following were tracked:

Training loss
Validation loss
Test loss

Loss curves were plotted to analyze:

convergence
overfitting
generalization
Sample Output

Example prediction:

Prediction: [0.0047, 0.1188, ..., 0.7910, ...]
Target:     [0, 0, 0, 1, 0, ...]

This demonstrates that the model predicts probability distributions instead of hard labels.

Results

Observations:

Training loss decreased steadily
Validation loss remained higher
Model successfully learned soft-label distributions
Slight overfitting observed

Example test loss:

Test Loss ≈ 1.27
Key Concepts Used
Soft Labels
Human Uncertainty Modeling
Probability Distributions
KL Divergence
Deep Learning
CNNs
ResNet-18
PyTorch Training Pipeline
Repository Structure
project/
│
├── data/
├── models/
├── notebooks/
├── outputs/
├── README.md
└── requirements.txt
How to Run
1. Clone Repository
git clone https://github.com/jcpeterson/cifar-10h
2. Install Dependencies
pip install torch torchvision numpy matplotlib
3. Run Notebook

Open the notebook in Google Colab and run all cells sequentially.

References
CIFAR-10H Official Repository
CIFAR-10H GitHub Repository
Human uncertainty makes classification more robust

CIFAR-10 Dataset Information

CIFAR-10H TensorFlow Dataset Documentation

Conclusion

This project demonstrates how deep learning models can be trained to capture human disagreement and uncertainty instead of predicting only hard labels. By using soft labels from CIFAR-10H and KL divergence loss, the model learns richer representations of image ambiguity and perceptual uncertainty.
