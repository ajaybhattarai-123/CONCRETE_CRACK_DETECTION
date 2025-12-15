# Concrete and Pavement Crack Detection System

> **Note to Users**: A detailed **[Documentation.pdf](file:///d:/PYTHON/CONCRETE_FINAL/Documentation.pdf)** is included in this repository. Please refer to it for in-depth theoretical background, methodology, and comprehensive usage instructions.

## Project Overview

This repository contains a deep learning solution for automated crack detection in concrete and pavement surfaces. Implemented using PyTorch, the Convolutional Neural Network (CNN) achieves **>99% training accuracy** and **~98% test accuracy**, demonstrating robustness and high precision for civil engineering inspections.

## Dataset Information

The model is trained on the **Concrete and Pavement Crack Images** dataset.

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/oluwaseunad/concrete-and-pavement-crack-images?select=Positive)
- **Author/Collector**: Omoebamije Oluwaseun (Nigerian Army University Biu, Borno State, Nigeria)
- **Total Dataset Size**: 30,000 images (15,000 Positive / 15,000 Negative)
- **Image Specifications**: 227 x 227 pixels, RGB format
- **Collection Method**: DJI Mavic 2 Enterprise drone (aerial) and smartphone (ground-level).

> **Important Implementation Note**: Due to hardware resource constraints for this demonstration, a stratified subset of **2,000 images** (1,000 Positive + 1,000 Negative) was used for training and testing. The model is fully capable of scaling to the full dataset with adequate compute resources.

## Key Features

- **Custom CNN Architecture**: Optimized for binary image classification with dynamic layer sizing.
- **High Performance**: Rapid convergence within 20 epochs using the Adamax optimizer.
- **Robustness**: Includes Dropout and Batch Normalization to prevent overfitting and ensure generalization.
- **Documentation**: Full theoretical and practical documentation provided in `Documentation.pdf`.

## Repository Structure

```
CONCRETE_FINAL/
├── concrete_pytorch.ipynb    # Complete implementation (Data loading, Model, Training, Evaluation)
├── Documentation.pdf         # Detailed project documentation and theory
└── README.md                 # Project overview and quick start guide
```

## Quick Start

1.  **Environment Setup**:
    Ensure you have Python 3.7+ installed along with the following libraries:
    - `torch`, `torchvision`
    - `numpy`, `matplotlib`, `PIL`

2.  **Data Preparation**:
    Download the dataset from the Kaggle link above. Organize it into `Positive_Analysis` and `Negative_Analysis` directories.

3.  **Running the Code**:
    Open `concrete_pytorch.ipynb` in Jupyter Notebook or Google Colab.
    - If using Google Colab, upload the dataset to Drive and mount it.
    - If running locally, simply update the path variables in the notebook.
    
    The notebook is self-documenting, with markdown cells explaining every step of the pipeline.

## Model Performance Summary

| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| **Accuracy** | ~100% | **98.33%** |
| **Loss** | 0.0012 | 0.1662 |

The low generalization gap (difference between train and test accuracy) indicates a well-regularized model that performs reliably on unseen data.

## Credits & Citation

This project utilizes data collected by **Omoebamije Oluwaseun**. If you use this dataset in your research or applications, please ensure proper citation as requested by the author.

---
*For a complete understanding of the system architecture, mathematical foundations, and detailed results discussion, please consult the **Documentation.pdf** file.*
