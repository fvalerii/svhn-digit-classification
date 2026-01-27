# 🖼️ SVHN Digit Classification: MLP & CNN Deep Learning Models
### *Neural Networks & Computer Vision Research*

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-SVHN-blueviolet.svg)

---

## 📋 Project Overview
This repository contains an end‑to‑end deep learning project for classifying digits from the **Street View House Numbers (SVHN)** dataset.  
Two models are implemented and compared:

- A **Multilayer Perceptron (MLP)**
- A **Convolutional Neural Network (CNN)**

The CNN achieves higher accuracy with **7× fewer parameters**, demonstrating the efficiency of convolutional architectures for image‑based tasks.

---

## 🎯 Research Objective
The SVHN dataset is significantly more challenging than MNIST because it contains real-world house numbers with varying backgrounds, lighting, and orientations. This project investigates the efficiency of spatial feature extraction by comparing a flat MLP architecture against a spatially-aware CNN.

---

## 📊 Dataset: SVHN

The **SVHN dataset** contains real‑world digit images extracted from Google Street View.  
It is more challenging than MNIST due to:

- Natural scene backgrounds  
- Varying lighting conditions  
- Cropped digits with noise  

Dataset reference:  
**Y. Netzer et al., “Reading Digits in Natural Images with Unsupervised Feature Learning,” NIPS Workshop, 2011.**

---

## 📊 Methodology 
The project was executed in a structured research pipeline:
* **Preprocessing:** Normalization of RGB images to grayscale and scaling pixel values to the $[0, 1]$ range to improve convergence speed.
* **MLP Architecture:** Developed a deep feed-forward network with Dropout layers to mitigate overfitting.
* **CNN Architecture:** Implemented a Convolutional pipeline with Max-Pooling and Flattening to capture local spatial hierarchies.
* **Optimization:** Utilized the Adam optimizer and Sparse Categorical Crossentropy loss.
* **Callbacks:** Implemented `EarlyStopping` and `ModelCheckpoint` to ensure optimal weights were retained and to prevent training stagnation.

---

## 📈 Results & Evaluation
The study confirms that the CNN architecture is substantially more robust for image-based tasks due to its ability to preserve local connectivity.

| Architecture | Test Accuracy | Observations |
| :--- | :--- | :--- |
| **MLP (Baseline)** | ~88% | Faster training, but struggled with noisy backgrounds. |
| **CNN (Champion)** | ~92% | Superior generalization and spatial awareness. |

### **Strategic Analysis: Loss & Accuracy**
The CNN demonstrated smoother convergence curves. Below the visualization, the model's ability to generalize is evident as the validation loss closely tracked the training loss, proving the effectiveness of the Dropout and BatchNormalization layers used.

![XGBoost Feature Importance](./images/xgb_feature_importance.png)

---

## 💡 Key Insights
* **Spatial Hierarchy:** Unlike MLPs, which treat pixels as independent features, CNNs successfully identified "edges" and "shapes," which are crucial for digit recognition.
* **Regularization:** Without Dropout layers, both models exhibited rapid overfitting, highlighting the importance of stochastic regularization in deep networks.
* **Real-world Application:** The CNN model proved reliable even with distorted digits and varying crop qualities inherent in the SVHN dataset.

---

## 📂 Project Deliverables
- **[Jupyter Notebook](./notebooks/svhn_image_classifier.ipynb):** Full end-to-end TensorFlow implementation, including data pipeline and model training.

---

## ⚙️ Installation & Setup
To replicate this research environment:

### 1. Clone the Repository
```bash
git clone https://github.com/fvalerii/svhn-image-classification.git
```
### 2. Install Required Python Packages
It is recommended to use a environment with Python 3.12.8 and GPU support:
##### Option A: Using Pip
```bash
pip install -r requirements.txt
```
##### Option B: Using Conda
```bash
conda env create -f environment.yml
conda activate salifort_research
```
### 3. Data Setup (Mandatory)
The SVHN dataset (format 2) consists of `.mat` files.
1. Download from: http://ufldl.stanford.edu/housenumbers/ (ufldl.stanford.edu)
2. Place the .mat files in the data/ directory
### 4. Run the Notebook
Open the Jupyter Notebook located at notebooks/svhn_image_classifier.ipynb using VS Code or JupyterLab.

---

## 💻 Tech Stack
- **Frameworks:** TensorFlow 2.x, Keras
- **Libraries:** NumPy, Scipy, Matplotlib, Pandas
- **Architecture:**Convolutional Neural Networks (CNN), Multi-Layer Perceptrons (MLP)

---

> **Note:** This project demonstrates mastery of Deep Learning workflows, including architecture design, callback implementation, and comparative model evaluation.
