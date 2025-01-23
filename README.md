# Brain Tumor Segmentation

## Overview
This project focuses on the segmentation of brain tumors using deep learning techniques. The goal is to accurately identify and segment tumor regions from brain MRI scans to aid in medical diagnosis and treatment planning.

## Features
- **Automated Tumor Segmentation:** Leverages deep learning models for precise segmentation.
- **Multi-Class Segmentation:** Differentiates between various tumor sub-regions.
- **Preprocessing Pipeline:** Includes data normalization, augmentation, and resizing.
- **Evaluation Metrics:** Utilizes Dice coefficient, IoU, and accuracy for performance assessment.
- **Visualization Tools:** Provides heatmaps and overlays for better interpretability.

## Dataset
The project uses the **BraTS (Brain Tumor Segmentation) dataset**, which includes:
- T1-weighted MRI
- T1c (contrast-enhanced T1-weighted MRI)
- T2-weighted MRI
- FLAIR (Fluid-Attenuated Inversion Recovery)

Ensure you have permission to use the dataset and comply with the data usage policies.

## Installation

To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Model Training

To train the model, run:

```bash
python src/train.py --epochs 50 --batch_size 16 --lr 0.001
```

Adjust hyperparameters as needed.

## Model Evaluation

To evaluate the trained model, run:

```bash
python src/evaluate.py --model_path models/best_model.pth
```

Evaluation results, including Dice score and confusion matrix, will be saved in the `results/` directory.

## Inference

To perform segmentation on new images:

```bash
python src/inference.py --input_image path/to/image.nii --output_path path/to/output
```

## Folder Structure
```
brain-tumor-segmentation/
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── models/
│   ├── checkpoints/
│   └── best_model.pth
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── requirements.txt
└── README.md
```

## Requirements
Ensure the following dependencies are installed:

- Python 3.8+
- TensorFlow/PyTorch
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- nibabel (for handling NIfTI files)

## Acknowledgements
This project is inspired by research on medical image analysis and utilizes publicly available datasets for non-commercial use.


## Contact
For any questions or collaborations, please reach out to:

**Email:** fitse.fani@gmail.com 
**LinkedIn:** [https://linkedin.com/in/yourprofile ](https://www.linkedin.com/in/fitsum-mesfin-25a01a185/) 
**GitHub:** https://github.com/fitsumM12

