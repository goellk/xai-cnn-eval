# xai-cnn-eval

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)

This repository contains the source code and the evaluation results presented in the paper "How a Model’s Architecture and Performance Influences Explainability: A Study of Grad-CAM and CRAFT in CNN-Based Image Classification".

---

## 📘 Overview

This repository provides:

- Training code for three convolutional neural network architectures:
  - **VGG16** [Simonyan & Zisserman, 2015]
  - **ResNet50** [He et al., 2016]
  - **ConvNeXt-T** [Liu et al., 2022]
- Implementations of two XAI method evaluation pipelines:
  - **Grad-CAM (Gradient-weighted Class Activation Mapping)** [Selvaraju et al., 2017]
  - **CRAFT (Concept Recursive Activation FacTorization)** [Fel et al., 2015]
- Results of the Grad-CAM and CRAFT evaluations.
- A `requirements.txt` file for setting up the Python environment.

Please note that you have to add datasets to the "Data" directory by yourself.

---

## 📁 Repository Structure

```
xai-cnn-eval/
│
├── Code/
│   ├── VGG16/
│   │   ├── training.py
│   │   ├── helper.py
│   │   └── [XAI method scripts]
│   │
│   ├── ResNet50/
│   │   ├── training.py
│   │   ├── helper.py
│   │   └── [XAI method scripts]
│   │
│   ├── ConvNeXt-T/
│       ├── training.py
│       ├── helper.py
│       └── [XAI method scripts]
│
├── Data/
│   └── [datasets]
│
├── Results/
│   ├── Grad-CAM/
│   └── CRAFT/
│
└── requirements.txt
```

---

## 🧩 Environment Setup

To reproduce the experiments and run the code, we recommend using **Python 3.11** in a virtual environment.

### 1. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

*(To be completed — usage examples and run commands will be added later.)*


---

## 🧠 Citation

If you use this repository or parts of it in your work, please cite the paper appropriately:

*(To be completed — paper citation will be added later.)*

---

## 📚 References

### Model Architectures

- VGG16: Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- ConvNeXt: Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s.

### XAI Methods

- Grad-CAM: Selvaraju, Ramprasaath R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.” 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626.
- CRAFT: Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien Colin, Rémi Cadène, and Thomas Serre. “CRAFT: Concept Recursive Activation Factori­zation for Explainability.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

---

## 📄 Acknowledgment
This work is part of the **XRAISE** research project by the  
**Deutsches Zentrum für Schienenverkehrsforschung (DZSF)** at the **Eisenbahn-Bundesamt**.
*(To be completed — further acknowledgements will be added later.)*

---
