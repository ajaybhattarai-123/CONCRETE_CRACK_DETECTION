# 🏗️ Concrete and Pavement Crack Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![CNN](https://img.shields.io/badge/CNN-Computer%20Vision-orange?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An intelligent, deep learning-based system for automated crack detection in civil infrastructure**

[Features](#-features) • [Installation](#-installation--setup) • [Usage](#-usage) • [Results](#-model-performance) • [Documentation](#-documentation)

</div>

---

## 📖 About The Project

This project utilizes a custom **Convolutional Neural Network (CNN)** to perform automated binary classification of concrete and pavement surfaces. It accurately distinguishes between "Positive" (cracked) and "Negative" (non-cracked) surfaces, providing a robust tool for infrastructure inspection and maintenance.

> **Note**: A detailed **[Documentation.pdf](./Documentation.pdf)** is included in this repository. Please refer to it for in-depth theoretical background, methodology, and comprehensive usage instructions.

---

## 📚 Documentation

For detailed theory, technical documentation, and comprehensive explanation of the code implementation, please refer to:

**[📄 Documentation.pdf](./Documentation.pdf)** - Complete guide covering:
- CNN architecture and theory
- Data preprocessing pipeline
- Implementation details and code explanation
- Mathematical foundations
- Experimental results and analysis

---

## ✨ Features

*   ✅ **High-Accuracy Detection**: Achieves **>98% accuracy** on the test set.
*   🧠 **Custom CNN Architecture**: Optimized specifically for binary image classification of structural surfaces.
*   🔄 **Data Preprocessing**: Includes automated resizing, tensor conversion, and normalization.
*   🛡️ **Robust Training**: Utilizes **Dropout** and **Batch Normalization** to prevent overfitting.
*   📉 **Real-time Monitoring**: Tracks loss and accuracy metrics during training for performance analysis.
*   ⚡ **Efficient**: Rapid convergence within 20 epochs using the Adamax optimizer.

---

## 🛠️ Requirements

| Requirement | Details |
|------------|---------|
| **Python Version** | 3.7 or higher |
| **Framework** | PyTorch |
| **Environment** | Jupyter Notebook / Google Colab |
| **GPU** | Recommended (CUDA supported) |

### Dependencies

The following libraries are required:

| Package | Purpose |
|---------|---------|
| `torch` | Deep Learning framework |
| `torchvision` | Image datasets and transforms |
| `PIL` | Image loading and processing |
| `numpy` | Numerical computations |
| `matplotlib` | Visualization |

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/concrete-crack-detection.git
cd concrete-crack-detection
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision numpy matplotlib pillow
```

### Step 3: Dataset Preparation

1.  Download the dataset from **[Kaggle](https://www.kaggle.com/datasets/oluwaseunad/concrete-and-pavement-crack-images?select=Positive)**.
2.  Extract the files into a directory.
3.  Organize folders as `Positive_Analysis` (cracked) and `Negative_Analysis` (non-cracked).

> **Important**: This project uses a subset of **2,000 images** (1,000 per class) for training demonstration due to hardware constraints.

---

## 🎮 Usage

### Running the System

Open the Jupyter Notebook to start the training and evaluation process:

```bash
jupyter notebook concrete_pytorch.ipynb
```

### Workflow

1.  **Mount Drive (Colab only)**: Connect to your dataset storage.
2.  **Load Data**: The system automatically preprocesses and splits data into train/test sets.
3.  **Train Model**: Run the training loop to optimize the CNN.
4.  **Evaluate**: Check performance metrics on the test set.
5.  **Visualize**: View loss/accuracy graphs and sample predictions.

---

## 📊 Model Performance

| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| **Accuracy** | ~100% | **98.33%** |
| **Loss** | 0.0012 | 0.1662 |

The low generalization gap indicates a well-balanced model that performs reliably on unseen data.

---

## 📂 Project Structure

```
concrete-crack-detection/
│
├── Positive_Analysis/        # Dataset: Images of cracked surfaces
├── Negative_Analysis/        # Dataset: Images of non-cracked surfaces
├── concrete_pytorch.ipynb    # Core implementation (Data loading, CNN Model, Training)
└── README.md                 # Project documentation
```
---
## 📄 License

This project is licensed under the MIT License.


---

## 🤝 Credits & Acknowledgments

*   **Dataset Author**: Omoebamije Oluwaseun (Nigerian Army University Biu)
*   **Platform**: Kaggle

If you use this dataset in your research, please ensure proper citation of the original author.

---


<div align="center">


⭐ **If this project helped you, please give it a star!** ⭐

</div>



