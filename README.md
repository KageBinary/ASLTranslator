# 🧠 ASL Translator – Real-Time Static and Dynamic Sign Recognition

This project is a real-time American Sign Language (ASL) translator that recognizes hand gestures from webcam input. It currently supports static ASL fingerspelling (A–Y), with future extensions planned for dynamic gestures and word-level recognition.

> ✅ This repository contains only code and documentation. All large files (models, datasets) have been excluded as required.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KageBinary/ASLTranslator.git
   cd ASLTranslator
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Run the real-time translator:
```bash
python src/pipeline/static_translator.py
```

### Collect training data for static signs:
```bash
python src/data_collectors/collect_static.py
```

### Update the dataset with new entries:
```bash
python scripts/update_master_csv.py
```

### Train the recognition model:
```bash
python scripts/train_static_model.py
```

---

## 📁 File Overview

```
ASLTranslator/
├── data/                # Raw and processed data (excluded from submission)
├── models/              # Trained models (excluded from submission)
├── scripts/             # Training and CSV update scripts
├── src/                 # Source code for collection, translation, detection
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── LICENSE              # License info
```

---

## 📊 Output

When the translator is running, a webcam window will show the input with live ASL letter predictions. Hold your hand steady to improve accuracy. Predictions are displayed in real time. Note: letters like **J** and **Z** are excluded due to their motion-based nature.

---

## 🧰 Dependencies

- `opencv-python` – for webcam capture
- `mediapipe` – for hand landmark detection
- `tensorflow` – for training and inference
- `pandas`, `numpy` – for data processing
- `scikit-learn` – for preprocessing and model evaluation

---

## 👥 Contributors

Ian Hock  
Nitin Chatlani  
Ana Krassowizki  
Nate Bomar  
Diego Lopez  
Pranaav Krishna Srinivasan

---

## 📚 References

- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) – Hand landmark detection API used for tracking.
- [TensorFlow](https://www.tensorflow.org/) – Deep learning framework used for model training.
- [scikit-learn](https://scikit-learn.org/stable/documentation.html) – Preprocessing, scaling, and evaluation tools.
- [OpenCV Documentation](https://docs.opencv.org/) – For webcam and image processing.
- [ASL Alphabet Reference](https://www.startasl.com/american-sign-language-alphabet/) – Used to guide label creation and validation.

