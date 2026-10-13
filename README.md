# Deepfake Detection System

A deep learning-based application for detecting manipulated images and videos. The system combines computer vision techniques, machine learning models, and visual explanations to help distinguish authentic content from AI-generated or manipulated media.

## Overview

The project explores different approaches to deepfake detection and compares their performance on a dataset containing real and fake images.

The application provides a web interface connected to a Python backend, allowing users to submit media for analysis and receive detection results.

## Key Features

- **Image detection:** classify submitted images as `REAL` or `FAKE`.
- **Video analysis:** examine video frames to identify potential manipulation.
- **Model comparison:** evaluate different machine learning and deep learning approaches.
- **Face analysis:** use MediaPipe for face detection and facial landmark extraction.
- **Texture analysis:** explore image texture features using GLCM-based methods.
- **Frequency analysis:** examine frequency-domain information to identify potential manipulation artifacts.
- **Explainability:** use Grad-CAM to visualize image regions that influence model predictions.
- **Web interface:** interact with the detection system through a Next.js application.

## Technologies

| Component        | Technologies               |
| ---------------- | -------------------------- |
| Frontend         | Next.js, React, TypeScript |
| Backend API      | Python, Flask, Flask-CORS  |
| Deep learning    | PyTorch, Torchvision, timm |
| Computer vision  | OpenCV, Pillow, MediaPipe  |
| Data analysis    | NumPy, SciPy, scikit-image |
| Machine learning | scikit-learn               |
| Testing          | pytest, pytest-cov         |

## Models Evaluated

The project compares three approaches:

- **EfficientNet-B0:** a convolutional neural network evaluated for deepfake classification.
- **ResNet18:** a residual neural network used for image classification.
- **Support Vector Machine (SVM):** a traditional machine learning baseline.

## Experimental Results

The academic evaluation reported the following results:

| Model           | Accuracy | Precision | Recall | F1-score | AUC-ROC |
| --------------- | -------: | --------: | -----: | -------: | ------: |
| SVM             |   77.20% |    77.40% | 76.90% |   77.10% |   0.855 |
| EfficientNet-B0 |   85.80% |    82.80% | 90.30% |   86.40% |   0.946 |
| ResNet18        |   88.88% |    89.10% | 88.60% |   88.80% |  0.9656 |

These values reflect the experiments documented in the academic report. They should not be interpreted as a guarantee of performance on unseen datasets or real-world media.

## Dataset

The experimental dataset contains **20,000 images**:

- 10,000 real images
- 10,000 fake images

The report describes an 80/20 training-validation split. The dataset draws on real and synthetic/manipulated image sources.

Due to repository size and dataset distribution considerations, the dataset is excluded from Git using `.gitignore`.

## Project Structure

```text
deep-learning-project/
├── app/                 # Frontend application pages
├── components/          # Reusable interface components
├── backend/             # Flask API
├── src/                 # Detection and processing modules
├── lib/                 # Frontend utilities
├── models/              # Model files (excluded from Git)
├── data/                # Dataset files (excluded from Git)
├── config.py            # Python configuration
├── main.py              # Python entry point
├── requirements.txt     # Python dependencies
├── package.json         # Frontend dependencies and scripts
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or later
- Node.js and npm
- Git

The project dependencies are pinned in `requirements.txt`. Compatibility with newer Python versions should be verified before installation.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd deep-learning-project
```

### 2. Set up the Python environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only PyTorch installations, consult the official PyTorch installation instructions.

### 3. Install frontend dependencies

```bash
npm install
```

### 4. Start the application

Start the Flask backend and Next.js frontend using the appropriate entry points and configuration for your local environment.

The required model weights must be available in the configured model directories before running inference. They are not included in this repository by default.

## Limitations

- Detection performance depends on the dataset, image quality, and manipulation techniques.
- Models may perform differently on content that was not represented during training.
- Video-level results depend on the frame sampling and temporal analysis procedures.
- Model weights and datasets are not distributed with the repository.

## Future Improvements

- Evaluate the system on additional independent datasets.
- Improve robustness against unseen manipulation techniques.
- Optimize inference time and memory consumption.
- Expand video analysis and temporal consistency evaluation.
- Improve model explanations and user-facing reports.

## Academic Context

This project was developed as part of an academic study of deepfake detection using machine learning, deep learning, and computer vision.

**Contributors:** Aya Bourhdifa, Douae Hajji and Hajar Tarfouss
