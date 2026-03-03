# VGG-CNN
# Evaluation of a 5-Stage VGG-Based Convolutional Neural Network for Multi-Class Glioma Classification

## 1. Abstract
This repository provides a PyTorch implementation of a deep convolutional neural network (CNN) optimized for automated diagnostic classification of brain MRI imagery. The model architecture is a modular reimplementation of the VGG-16 design philosophy, specifically tailored to handle the high-variance spatial features of neurological pathology.

## 2. Architectural Design & Rationale
The model utilizes 10 convolutional layers organized into 5 discrete feature-extraction blocks.

* **Convolutional Motif:** Adheres to the $3 \times 3$ kernel standard to increase the receptive field while minimizing parameter overhead.
* **Sequential Non-Linearity:** Each block contains dual convolutional operations followed by ReLU activation, facilitating the extraction of complex hierarchical features (e.g., tumor margins, necrotic centers).
* **Spatial Downsampling:** Employs $2 \times 2$ Max-Pooling with a stride of 2, reducing the input resolution from $224^2$ to $7^2$ across 5 stages.
* **Global Aggregation:** Utilizes `AdaptiveAvgPool2d(1, 1)` to generate a 64-dimensional feature vector, mitigating the risk of overfitting associated with high-dimensional fully connected layers in the original VGG specification.



## 3. Model Specification (PyTorch)

| Layer Stage | Input Depth | Output Depth | Resolution Change |
| :--- | :--- | :--- | :--- |
| Block 1 | 3 | 8 | $224 \rightarrow 112$ |
| Block 2 | 8 | 16 | $112 \rightarrow 56$ |
| Block 3 | 16 | 32 | $56 \rightarrow 28$ |
| Block 4 | 32 | 64 | $28 \rightarrow 14$ |
| Block 5 | 64 | 64 | $14 \rightarrow 7$ |
| Global Pool | 64 | 64 | $7 \rightarrow 1$ |
| Classifier | 64 | 4 | Final Logits |

## 4. Training & Optimization Metrics
The training process was executed using the following hyperparameters:
* **Optimization Algorithm:** Adam ($lr = 0.001$)
* **Loss Criterion:** Categorical Cross-Entropy
* **Batch Size:** 32 (Adjustable based on VRAM)

### Observed Convergence Log:
| Epoch | Step | Loss | Improvement |
| :--- | :--- | :--- | :--- |
| 1 | 90/90 | 1.3395 | Baseline |
| 5 | 90/90 | 0.7908 | -41.0% |
| 10 | 90/90 | 0.6876 | -49.2% |



## 5. Requirements
To reproduce these results, ensure the following dependencies are installed:
* `torch`
* `torchvision`
* `numpy`
* `matplotlib` (for loss visualization)
