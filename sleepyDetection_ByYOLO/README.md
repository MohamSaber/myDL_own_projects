# 🚗 Driver Alert System (YOLO-Based Driver Monitoring)

A real-time driver monitoring system that detects unsafe driver behaviors using a **YOLO-based object detection model**. The system identifies actions such as **phone usage, drowsiness, distracted driving, and other risky behaviors**, then issues alerts to improve road safety.

---

## 📌 Features

* Real-time driver behavior detection
* Detects phone usage and driver distraction
* Drowsiness monitoring
* YOLO-based deep learning model
* Video inference support
* Modular and easy-to-extend codebase

---

## 📂 Project Structure

```text
Driver-Alert-System/
├── notebooks/
│   ├── train.ipynb          # Model training
│   └── inference.ipynb      # Model testing
│
├── src/
│   ├── api_loader.py
│   ├── detect.py            # Inference script
│   └── utils.py
│
├── configs/
│   ├── data.yaml
│   └── model.yaml
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Requirements

* Python 3.9+
* Install dependencies:

```bash
pip install -r requirements.txt
```

### Main Libraries

* TensorFlow
* Ultralytics (YOLO)
* OpenCV
* NumPy
* Pygame

---

## 🚀 Getting Started

### Train the Model

Open and run:

```bash
notebooks/train.ipynb
```

### Run Inference

```bash
python src/detect.py --source path/to/video.mp4 --weights path/to/best.pt
```

---

## 📊 Results

The model is capable of detecting multiple driver behaviors, including:

* 📱 Phone usage
* 😴 Drowsiness
* 🚫 Distracted driving
* ⚠️ Unsafe actions

The system has been evaluated on multiple driving videos and demonstrates reliable real-time performance for driver monitoring applications.

---

## 📥 Resources

**Test Videos**

> Google Drive: `ADD_LINK_HERE`

**Trained Model Weights**

> Google Drive: `ADD_LINK_HERE`

---

## 📝 Notes

* Large assets such as `.pt` weights and `.mp4` videos are excluded from this repository.
* Download them using the links above.
* `.gitignore` excludes generated files including:

  * `runs/`
  * `results/`
  * `*.pt`
  * `*.mp4`
  * virtual environments

---

## 🛠️ Tech Stack

* Python
* YOLO (Ultralytics)
* TensorFlow
* OpenCV
* NumPy
* Pygame

---

## 👨‍💻 Author

**Mohamed Saber**

Computer Engineering Graduate | Machine Learning & Deep Learning Engineer
