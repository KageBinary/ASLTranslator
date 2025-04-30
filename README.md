# ASL Translator: Static, Dynamic, and Word Recognition

This project enables real-time translation of American Sign Language (ASL) via webcam. It currently supports static letter recognition (A–Y) and is expanding toward dynamic signs and full word recognition.

---

## 🛠️ Components and What They Do

| File/Folder | Purpose |
|:---|:---|
| `src/data_collectors/` | 📸 Scripts to collect hand pose data for static (letter) and dynamic (motion-based) signs. |
| `src/models/` | 🧠 Defines neural network architectures for letters and word sequences. |
| `src/pipeline/` | 👥 Pipelines for real-time translation: static, dynamic, and word-level. |
| `src/preprocessing/` | 💡 Tools for data augmentation and sequence extraction. |
| `src/utils/hand_detector.py` | ✋ Detects hands and extracts features using MediaPipe. |
| `scripts/` | 📆 Training and evaluation scripts for static letters and words. |
| `data/` | 📂 Raw, processed, and sequence data storage. |
| `models/` | 🌐 Saved trained models and scalers. |
| `app.py` | 📅 Main launcher (for future app integration). |

---

## 🛆 Libraries Used

- **OpenCV** (`cv2`) — Webcam capture and processing
- **MediaPipe** — Hand landmark detection
- **NumPy** — Numerical operations
- **Pandas** — CSV and data handling
- **Joblib** — Saving models and scalers
- **Scikit-learn** (`sklearn`) — Data scaling, splitting
- **TensorFlow / Keras** — Deep learning models
- **Matplotlib, Seaborn** — Data visualization
- **Jupyter** — Notebook experiments

---

## 🔥 Workflow

### Static Letters (Fingerspelling)

1. **Data Collection**
   ```bash
   python src/data_collectors/collect_static.py
   ```

2. **Update Master Dataset**
   ```bash
   python scripts/update_master_csv.py
   ```

3. **Train Static Model**
   ```bash
   python scripts/train_static_model.py
   ```

4. **Live Static Translator**
   ```bash
   python src/pipeline/static_translator.py
   ```

### Dynamic Signs (In Progress)
- Collect motion-based sequences.
- Train sequence models (`train_word_model.py`).
- Run dynamic sign translator (`dynamic_translator.py`).

### Word-Level Translation (In Progress)
- Combine static and dynamic predictions.
- Translate sequences of letters or gestures into full words (`word_translator.py`).

---

## 📆 Directory Structure

```
ASLTranslator/
|
|├— data/
|   ├— processed/ —> Processed CSVs
|   ├— raw/ —> Raw capture data
|   ├— reference/ —> Reference images
|   └— sequences/ —> Dynamic sequence captures
|
|├— models/
|   ├— static/ —> Static letter models
|   └— dynamic/ —> (planned) Dynamic models
|
|├— src/
|   ├— data_collectors/
|   ├— models/
|   ├— pipeline/
|   ├— preprocessing/
|   ├— utils/
|
|├— scripts/
|
|├— notebooks/
|
|├— app.py
|├— LICENSE
|├— README.md
|└— requirements.txt
```

---

## ✨ Features

- 💬 Real-time static ASL letter translation
- 💡 Data augmentation and sequence extraction for dynamic signs
- 🧬 Early stopping and dynamic learning rate during training
- 🔥 Light and fast model execution
- 📢 Word-level translation planned!

---

## 📌 Notes

- Letters **J** and **Z** are excluded from static because they involve motion.
- Good lighting and centered hands improve accuracy.
- Future expansion includes **dynamic sequence detection** and **full word formation**!

---

## 🎉 Future Goals

- Train dynamic sequence models
- Integrate dynamic and static recognizers
- Build a simple web or desktop app for users
- Open source the full codebase with pretrained models!

---

