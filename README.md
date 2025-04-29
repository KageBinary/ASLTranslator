# ASL Static Translator (Fingerspelling Recognition)

This project allows real-time translation of American Sign Language (ASL) static alphabet signs (A–Y, excluding J and Z) using a webcam.

---

## 🛠️ Components and What They Do

| File | Purpose |
|:---|:---|
| `src/data/collect_static.py` | 📸 Collects landmark data of hand poses for each ASL letter (skipping J and Z) via webcam and saves them as CSVs. |
| `scripts/update_master_csv.py` | 🗂️ Combines all collected CSVs into a single master dataset `training_data_letters_MASTER.csv` for model training. |
| `scripts/train_static_model.py` | 🧠 Trains a TensorFlow neural network model on the collected landmark data and saves the model + feature scaler. |
| `src/pipeline/static_translator.py` | 🎥 Runs a live webcam app that detects your hand sign and types letters based on consistent predictions. |
| `src/hand_detector.py` | ✋ Wrapper around MediaPipe Hands to detect hands and extract normalized 3D landmarks and extra features for ML. |

---

## 🛆 Libraries Used

- **OpenCV** (`cv2`) — Webcam capture and display
- **MediaPipe** — Hand landmark detection
- **NumPy** — Mathematical operations
- **Pandas** — Data handling
- **Joblib** — Model and scaler saving
- **Scikit-learn** (`sklearn`) — Feature scaling and dataset splitting
- **TensorFlow / Keras** — Deep learning model training
- **glob, os, sys** — File operations

---

## 🔥 Workflow (Step-by-Step)

1. **Data Collection** (`collect_static.py`)
   - Collects 91 features from hand landmarks for each letter.
   - Saves them into session CSVs under `data/processed/`.

2. **Master Dataset Generation** (`update_master_csv.py`)
   - Merges all session CSVs into `training_data_letters_MASTER.csv`.

3. **Model Training** (`train_static_model.py`)
   - Loads master CSV, splits data, scales features.
   - Trains a deep neural network model.
   - Saves trained model (`letter_model.h5`) and scaler (`feature_scaler.pkl`).

4. **Live Translation** (`static_translator.py`)
   - Opens webcam.
   - Detects hand, extracts features.
   - Predicts letter.
   - Types letters into a text string with stability checks.
   - Save or clear text easily with keyboard shortcuts.

---

## 📆 Directory Structure

```
ASLTranslator/
|
|├— data/
|   └— processed/
|       ├— training_data_letters_*.csv
|       └— training_data_letters_MASTER.csv
|
|├— models/
|   ├— feature_scaler.pkl
|   └— letter_model.h5
|
|├— scripts/
|   ├— collect_static.py
|   ├— static_translator.py
|   ├— train_static_model.py
|   └— update_master_csv.py
|
└— src/
    └— hand_detector.py
```

---

## 🧪 How to Run

1. **Install required libraries**:
```bash
pip install opencv-python mediapipe numpy pandas scikit-learn tensorflow joblib
```

2. **Collect data**:
```bash
python scripts/collect_static.py
```

3. **Update master dataset**:
```bash
python scripts/update_master_csv.py
```

4. **Train the model**:
```bash
python scripts/train_static_model.py
```

5. **Run live translator**:
```bash
python scripts/static_translator.py
```

---

## ✨ Features

- ✋ Hand tracking and pose normalization
- 🔥 Real-time letter typing based on stable predictions
- 🚀 Light and efficient model suitable for laptop use
- 🔗 Text saving, clearing, spacing, and backspacing

---

## 📌 Notes

- **Letters J and Z are skipped** (they involve motion).
- Best results need **good lighting** and **clear hand visibility**.
- You can adjust `detection_confidence` inside `HandDetector` if detection isn't stable.

---

