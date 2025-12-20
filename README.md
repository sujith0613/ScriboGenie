# ScriboGenie — Handwriting Recognition System

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

Ensure the pretrained model `myCnn.h5` is in the `models` folder.

---

## Usage

Run the application:

```bash
python handwriting_app_modeA.py
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

---

## Demo Video

Watch the application in action:

[👉 Demo Video (Google Drive)](https://drive.google.com/file/d/1HfVmacgq0bILBhl5SYdO46YtjSfA_c2i/view)

---

## Project Structure

```text
ScriboGenie/
├── models/
│   └── myCnn.h5
├── handwriting_app_modeA.py
├── train_emnist_cnn.py
├── debug_pro/
└── README.md
```

---

## License

This project is intended for educational and research purposes only.  
You may use, study, and modify the code for academic or personal projects.  
Commercial use requires explicit permission from the authors.
