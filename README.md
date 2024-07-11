# AI-Based Alzheimer's Disease Diagnostic Tool from MRI Images

## Introduction

This project aims to develop an AI-powered tool to assist in the diagnosis of Alzheimer's disease from MRI images. The tool will help doctors and medical professionals analyze brain MRI scans more quickly and accurately, leading to earlier and more precise diagnoses of Alzheimer's.

## Project Progress

### Phase 1: Data Exploration and Preprocessing (Completed)

- Loaded and explored the "Alzheimer MRI Disease Classification Dataset" from Hugging Face.
- Preprocessed the data:
  - Converted images to RGB format (if necessary).
  - Resized images to 224x224 pixels.
  - Normalized pixel values using ImageNet mean and standard deviation.

### Phase 2: Model Building and Training (In Progress)

- Selected ResNet50 pre-trained on ImageNet as the base model.
- Modified the final layer for 4 classes (Non Demented, Very Mild Demented, Mild Demented, Moderate Demented).
- Implemented training loop with Adam optimizer and cross-entropy loss.
- Evaluated model performance on validation set and logged training metrics.
- Visualized training and validation metrics using Altair.

### Phase 3: Deployment and Real-World Evaluation (Not Started)

- Developing a user interface for the tool (using Flask).
- Deploying and testing the tool in medical facilities.
- Gathering feedback and evaluating the tool's real-world effectiveness.

## Installation

To set up the project environment and install the required packages, follow these steps:

1. **Clone the Repository:**
2. **Create a Virtual Environment (Recommended):**

- python -m venv venv
- source venv/bin/activate # On Windows, use `venv\Scripts\activate`

3. **Install Dependencies:**

- pip install -r requirements.txt

## Technologies Used

- Programming Language: Python
- Deep Learning Libraries: PyTorch
- Image Processing Libraries: OpenCV, Pillow
- Other Libraries: NumPy, Matplotlib, Hugging Face Datasets, tqdm, pandas, altair

## Dataset

**Alzheimer MRI Dataset**

- **Author:** Falah.G.Salieh
- **Year:** 2023
- **Version:** 1.0
- **Source:** Hugging Face
- **URL:** https://huggingface.co/datasets/Falah/Alzheimer_MRI

## Contact

For any questions or feedback, please contact us at:
