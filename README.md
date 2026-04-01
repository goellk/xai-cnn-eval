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
  - **CRAFT (Concept Recursive Activation FacTorization)** [Fel et al., 2023]
- Results of the Grad-CAM and CRAFT evaluations.
- A `requirements.txt` file for setting up the Python environment.

Please note that you have to add datasets to the "Data" directory by yourself.

---

## 📁 Repository Structure

```
xai-cnn-eval/
│
├── XAI_evaluation/
│   ├── craft/
|   |   └── readme.txt
│   └── gradcam/
|       ├── evaluate_gradcam_quantus_convnext-t.py
│       ├── evaluate_gradcam_quantus_resnet50.py
|       └── evaluate_gradcam_quantus__vgg16.py
|
├── training/
│   ├── train_convnext.py
│   ├── train_resnet.py
│   └── train_vgg.py
│
├── datasets/
│   ├── imagenet80_subset_classes.txt
│   └── split_imagenet.sh
│
├── results/
│   ├── Grad-CAM/GRADCAM.csv
│   └── CRAFT/CRAFT.csv
│
├── requirements.txt
|
└── README.md
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

1. **Dataset Preparation:**
   *(To be completed ...)*

2. **Model Training:**
   
   *(To be completed ...)*

3. **XAI Evaluation:**
   
   As described in our paper, we applied and evaluated the two XAI methods Grad-CAM and CRAFT in order to explain the model’s predictions and evaluate the explanations. For applying the scripts mentioned below the trained models are required. All scripts are available in the `XAI_evaluation` directory.
   
   - **Grad-CAM:** The XAI evaluation for Grad-CAM is implemented in the script `Grad-CAM/[convnext/resnet/vgg]/evaluate_gradcam_quantus.py`.
     
     Adjust the model and dataset paths and run the script:
     
     ```bash
     python evaluate_gradcam_quantus.py
     ```
   
   - **CRAFT:** For running and evaluating CRAFT on a specific model, you need to run two different scripts, located in `CRAFT/[convnext/resnet/vgg]` :
     
     ```bash
     python dump_craft_explanations.py
     ```
     
     and
     
     ```bash
     python evaluate_craft_explanations.py
     ```
     
     **Disclaimer:** Please note that the code associated with the CRAFT evaluation is partially unavailable, as the authors are currently working on publishing it in a separate paper. The link to the separate paper and repository will be added here soon.



---

## 🧠 Citation

If you use this repository or parts of it in your work, please cite the paper appropriately:

```
@inproceedings{xaieval2026,
  author    = {Kilian Göller and Haadia Amjad and Steffen Seitz and Carsten Knoll and Ronald Tetzlaff},
  title     = {How a Model’s Architecture and Performance Influences Explainability: A Study of Grad-CAM and CRAFT in CNN-Based Image Classification},
  booktitle = {Explainable Artificial Intelligence},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  note      = {To appear}
}
```

*(Paper accepted for presentation at the 4th XAI World Conference in Fortaleza, Brazil. Soon to be published.)*

---

## 📚 References

### Model Architectures

- VGG16: Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- ConvNeXt: Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s.

### XAI Methods

- Grad-CAM: Selvaraju, Ramprasaath R., et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.” 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626.
- CRAFT: Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien Colin, Rémi Cadène, and Thomas Serre. “CRAFT: Concept Recursive Activation Factori­zation for Explainability.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

### Evaluation Frameworks

- Quantus: Hedström, Anna, et al. “Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond.” Journal of Machine Learning Research, 2023.
- *(CRAFT Evaluation Framework To be added)*

---

## 📄 Acknowledgment

This work was supported by DFG (grant number TE257/37-1), by the Konrad Zuse School, SECAI (BMBF Project Nr. 57616814) and Deutsches Zentrum für Schienenverkehrsforschung (DZSF) at the Eisenbahn-Bundesamt as part of the XRAISE research project.

---
