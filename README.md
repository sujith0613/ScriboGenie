# ScriboGenie — Handwriting Recognition System

## Abstract

ScriboGenie is an assistive multisensory handwriting support system designed for children with learning disabilities such as dyslexia, dysgraphia, ADHD, and fine-motor coordination issues. Unlike traditional writing tools, ScriboGenie integrates digital handwriting capture, machine learning–based real-time recognition, and multi-sensory feedback mechanisms to support guided writing and learning.

The system uses a Wacom writing pad to capture pen strokes, which are processed using Raspberry Pi and analyzed using TensorFlow-based handwriting recognition models. A dyslexia-aware correction engine identifies common inversion and mirroring errors. Visual, auditory, and tactile feedback improves clarity, writing fluency, and confidence. This inclusive, adaptive tool aims to enhance the writing abilities of children requiring special educational support.

---

## Introduction

Children with specific learning disabilities (SLD) face challenges in handwriting, letter formation, spatial alignment, and fine-motor control. Dyslexia affects letter recognition and sequencing, dysgraphia affects handwriting clarity, and motor coordination disorders impact handwriting stability. These issues cause frustration, low confidence, and reduced academic performance.

ScriboGenie addresses these challenges by combining assistive technology and machine intelligence into a child-friendly writing system. The project incorporates:

- A digital writing surface  
- Real-time handwriting capture  
- ML-based handwriting recognition  
- Dyslexia-aware auto-correction  
- Speech support for students with writing difficulty  
- Feedback mechanisms to guide handwriting improvement  

This solution enables inclusive learning and supports foundational literacy development.

---

ScriboGenie is a Python-based handwriting recognition system with real-time character recognition and dyslexia-aware correction. The application provides a GUI for drawing, erasing, and live prediction, making it suitable for handwriting practice, testing, and educational applications.

---

## Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model](#model)  
- [Demo Video](#demo-video)  
- [Project Structure](#project-structure)  
- [License](#license)  

---

## Features

- Real-time recognition of handwritten characters (CNN-based)  
- Automatic segmentation and word grouping  
- Dyslexia-aware correction for common character confusions  
- Tkinter GUI with drawing, erasing, undo/redo, and adjustable brush/eraser  
- Offline functionality, no internet required  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sujith0613/ScriboGenie.git
cd ScriboGenie
```

Install dependencies:

```bash
pip install tensorflow pillow opencv-python spellchecker albumentations matplotlib
```

Ensure the pretrained model `myCnn.h5` is present in the project directory.

---

## Usage

Run the application:

```bash
python app9.py
```

Draw on the canvas; recognized and corrected text appears in the side panel.

Enable live prediction to update recognition in real-time.

---

## Model

CNN trained on EMNIST ByClass dataset (62 classes: digits, uppercase, lowercase).

Includes preprocessing:
- Orientation correction  
- Normalization  
- Resizing  

Optional data augmentation:
- Rotation  
- Scaling  
- Brightness / noise  

Trained model saved as `myCnn.h5`.

Training notebook:
- `CNN_Training.ipynb`

Evaluation artifact:
- `confMat_emnist.png`

---

## Demo Video

Watch the application in action:

[👉 Demo Video (Google Drive)](https://drive.google.com/file/d/1HfVmacgq0bILBhl5SYdO46YtjSfA_c2i/view)

---

## Project Structure

```text
ScriboGenie/
├── CNN_Training.ipynb
├── app9.py
├── myCnn.h5
├── confMat_emnist.png
└── README.md
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


