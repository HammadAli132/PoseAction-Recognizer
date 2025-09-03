# 🥊 PoseAction-Recognizer

PoseAction-Recognizer is a research-driven project for **boxing action recognition** using **computer vision** and **deep learning**.  
The system combines **object detection (YOLOv8)** and **pose estimation (skeleton-based models)** to classify different boxing punches and movements.  

---

## ⚠️ Project Status

🚧 **This project is currently under active development.**  
The repository structure (especially inside the `/data` directory) may undergo **major updates** as the project evolves.  
Expect changes in preprocessing pipelines, dataset formats, and training workflows as new functionality is added.  

---

## 🚀 Project Goals

- Automate **recognition of boxing punches**: jab, cross, hook, uppercut, etc.  
- Leverage **pose estimation (skeletons)** instead of relying only on bounding boxes.  
- Provide a foundation for future systems that can **predict counter-moves** during boxing training.  
- Support both **custom raw video datasets** and **pre-annotated datasets (e.g., Roboflow)**.  
- Offer clean, modular **data preprocessing, training, and visualization tools**.  

---

## 📂 Repository Structure

```bash
PoseAction-Recognizer/
│── data/                     # (ignored in .gitignore, must be created locally)
│   ├── raw/                  # Raw boxing videos organized by class
│   │   ├── jab/video1.mp4
│   │   ├── cross/video2.mp4
│   │   └── ...
│   │
│   ├── processed/
│   │   ├── frames/           # Extracted frames from videos
│   │   │   ├── jab_video1/frame_000001.jpg
│   │   │   └── metadata.json # FPS, frame count, etc. for each video
│   │   └── poses/            # Poses extracted from frames
│   │       ├── jab_video1/poses.json
│   │       ├── jab_video1/poses.npy
│   │       └── ...
│   │
│   ├── datasets/
│   │   └── roboflow/         # Boxing dataset from Roboflow
│   │       ├── train/images/...
│   │       ├── train/labels/...
│   │       ├── valid/images/...
│   │       ├── test/images/...
│   │       └── data.yaml     # YOLO dataset config
│   │
│   └── poses_datasets/       # Poses extracted from Roboflow dataset
│       ├── train/poses.json, poses.npy
│       ├── valid/poses.json, poses.npy
│       └── test/poses.json, poses.npy
│
│── notebooks/                # Jupyter notebooks for prototyping
│
│── scripts/                  # Utility scripts (standalone)
│   ├── prepare_data.py       # Extract frames from videos
│   ├── extract_poses.py      # Extract poses (from frames or datasets)
│   ├── convert_to_coco.py    # (planned) Convert YOLO+poses → COCO for CVAT
│   ├── train.sh              # Run training loop
│   └── evaluate.sh           # Evaluate model performance
│
│── src/                      # Main source code
│   ├── data/
│   │   └── preprocess.py     # CLI menu to run frame/pose extraction
│   ├── models/               # ML/DL models (MLP, LSTM, ST-GCN)
│   ├── training/             # Training pipelines
│   └── inference/            # Inference/deployment scripts
│
│── experiments/              # Logs, checkpoints, results
│── config.yaml               # Config file for paths & hyperparams
│── requirements.txt          # Python dependencies
│── README.md                 # You are here
```

---

## ⚙️ Functionality

### 🔹 1. Data Preprocessing
- **Video → Frames**  
  - Extract frames at a chosen FPS  
  - Save to `data/processed/frames/{action_video}/`  
  - Update `metadata.json` with fps + total frame count  

- **Frames → Poses**  
  - Run YOLOv8-pose on each frame  
  - Save skeletons as `poses.json` (readable) and `poses.npy` (fast loading)  

- **Dataset Images → Poses**  
  - Works with Roboflow dataset (`train/valid/test/images`)  
  - Extracts skeletons for every image  
  - Saves in `data/poses_datasets/{split}/poses.json, poses.npy`  

### 🔹 2. Model Training
- **YOLOv8 Detector**  
  - Trains directly on Roboflow dataset with bounding boxes  
  - Evaluates with mAP, precision, recall  

- **Pose Classifier**  
  - Inputs: extracted skeletons (`poses.npy`)  
  - Outputs: punch class (cross, jab, hook, uppercut, etc.)  
  - Models: MLP, RNN/LSTM, or Graph-based (future)  
  - Evaluates with accuracy, F1, confusion matrix  

- **Fusion (planned)**  
  - Combine bounding box detector + pose classifier predictions  

### 🔹 3. Visualization
- **Local visualization** (planned): overlay bounding boxes + skeletons on dataset images  
- **CVAT support**:  
  - Convert YOLO labels + pose annotations to **COCO keypoints JSON**  
  - Upload into CVAT → visualize bounding boxes + skeleton overlays together  

---

## 📊 Dataset Details

### Roboflow Dataset (BoxingHub)
- Classes: `['bag', 'cross', 'hook', 'jab', 'no punch', 'uppercut']`  
- Format: YOLOv8 (images + `.txt` annotations)  
- Provided splits: `train`, `valid`, `test`  
- License: CC BY 4.0  

### Custom Dataset
- Raw boxing videos placed in `data/raw/`  
- Can be automatically segmented into frames and poses  

---

## 🔧 Installation

```bash
# Clone repo
git clone https://github.com/<your-username>/PoseAction-Recognizer.git
cd PoseAction-Recognizer

# Create environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scriptsctivate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Extract Frames from Video
```bash
python -m src.data.preprocess
# Choose option 1
```

### Extract Poses from Frames
```bash
python -m src.data.preprocess
# Choose option 2
```

### Extract Poses from Roboflow Dataset
```bash
python -m src.data.preprocess
# Choose option 3
```

### Train YOLOv8 Detector (on Roboflow dataset)
```bash
yolo detect train data=data/datasets/roboflow/data.yaml model=yolov8s.pt epochs=50 imgsz=640
```

### Train Pose Classifier (planned)
```bash
python -m src.training.train_pose
```

---

## 📈 Evaluation

- **YOLO Detector** → mAP@50-95, per-class precision/recall  
- **Pose Classifier** → Accuracy, F1-score, confusion matrix  
- **Fusion** → Compare individual vs combined models  

---

## ✅ Roadmap
- [x] Video → Frames → Poses pipeline  
- [x] Support Roboflow dataset with images + annotations  
- [ ] Train pose classifier  
- [ ] Add fusion model for joint prediction  
- [ ] Visualization script for dataset inspection  
- [ ] Convert poses+labels → COCO keypoints for CVAT  

---

## 👨‍💻 Authors
- Hammad Ali & Team – Final Year Project  

---

## 📜 License
This project is released under the MIT License.  
Dataset (BoxingHub) is under **CC BY 4.0** (see Roboflow dataset page).  
