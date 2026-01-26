# 🖼️ SVHN Digit Recognition: A Deep Learning Comparative Study
### *Neural Networks & Computer Vision Research*

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Model](https://img.shields.io/badge/Model-XGBoost-red)
![Tech Stack](https://img.shields.io/badge/Tech%20Stack-Python%20%7C%20Pandas%20%7C%20Sklearn-orange)

---

## 📋 Project Overview
This research project focuses on the development of an image classification system for the Street View House Numbers (SVHN) dataset. The goal was to design, train, and evaluate two distinct neural network architectures—a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN)—to accurately identify house numbers in real-world images.

---

## 🎯 Business Objective
The SVHN dataset is significantly more challenging than MNIST because it contains real-world house numbers with varying backgrounds, lighting, and orientations. This project investigates the efficiency of spatial feature extraction by comparing a flat MLP architecture against a spatially-aware CNN.

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
- **[Project PDF Summary:](./docs/svhn_project_summary.pdf):** A comprehensive breakdown of the architecture, hyperparameters, and final performance metrics.

---

## ⚙️ Installation & Setup
To replicate this research environment:
### 1. Clone the Repository
```bash
git clone https://github.com/fvalerii/svhn-image-classification.git
```
### 2. Install Required Python Packages
It is recommended to use a GPU-enabled environment.
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
The dataset is hosted on Kaggle. To replicate this project:
1. Download the dataset **HR_comma_sep.csv** from [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction).
2. Place **HR_comma_sep.csv** inside the`/data/` folder of this project.
### 4. Run the Analysis
Open `notebooks/schn_image_classifier.ipynb` to view the training logs and model predictions.
---

## 💻 Tech Stack
- **Frameworks:** TensorFlow 2.x, Keras
- **Libraries:** umPy, Scipy (for .mat file loading), Matplotlib (Visualizations)
- **Data::** Street View House Numbers (SVHN) Dataset

---

> **Note:** This project demonstrates mastery of Deep Learning workflows, including architecture design, callback implementation, and comparative model evaluation.
