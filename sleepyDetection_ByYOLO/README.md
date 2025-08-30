

---

```markdown
# 🚗 Driver Alert System (YOLO-based)

This project focuses on monitoring **driver behavior** (phone usage, drowsiness, unsafe actions) using a **YOLO model** for action recognition inside the car. The system provides alerts when dangerous activities are detected.

---

## 📂 Project Structure
```
```bash
projLV/ (or renamed folder)
├── notebooks/           # Jupyter notebooks for training/testing
│    ├── train.ipynb
│    └── inference.ipynb
│
├── src/                 # Core Python scripts
│    ├── api\_loader.py
│    ├── detect.py
│    └── utils.py
│
├── configs/             # YOLO configuration files
│    ├── data.yaml
│    └── model.yaml
│
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
└── .gitignore           # Ignore unnecessary files

````

---

## ⚙️ Requirements
- Python 3.9+
- Install dependencies:
```bash
pip install -r requirements.txt
````

Main libraries:

* `tensoeflow`
* `ultralytics`
* `opencv-python`
* `numpy`
* `pygame`

---

## 🚀 Usage

### 1. Training

Run:

```bash
notebooks/train.ipynb
```

### 2. Inference

Run the model on a video:

```bash
python src/detect.py --source path/to/video.mp4 --weights path/to/best.pt
```

---

## 📊 Results

* The model can detect driver actions such as texting, talking on the phone, drowsiness, and more.
* Tested on multiple driving videos.

📥 **Download test videos:** [Google Drive Link](PUT_YOUR_LINK_HERE)
📥 **Download trained weights:** [Google Drive Link](PUT_YOUR_LINK_HERE)

---

## 📝 Notes

* Large files (videos `.mp4`, model weights `.pt`) are not included in the repository.
* Download them from the provided links above.
* `.gitignore` excludes `env/`, `runs/`, `results/`, `*.mp4`, `*.pt`, etc.

---

## 👨‍💻 Author

* **Mohamed Saber** — Machine Learning & Deep Learning Enthusiast

```

