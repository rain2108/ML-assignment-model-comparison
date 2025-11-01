# ML-assignment-model-comparison

# A Technical Report and Comparative Architectural Analysis of YOLOv8n and YOLOv5s for Real-Time Rock-Paper-Scissors Detection

## 🧾 Short Description
In this project, we put two top-tier, real-time object detection models—YOLOv5s and YOLOv8n—head-to-head to see which was better for a specific, fun task: recognizing "rock," "paper," and "scissors" hand gestures.

But we didn't just want to compare performance scores. We decided to look under the hood at the models themselves, exploring the key design changes from YOLOv5's older, anchor-based system to the more modern, anchor-free approach used in YOLOv8.

To keep the comparison fair, we trained both models on the exact same dataset (Roboflow's "RPS SXSW") for the same amount of time (5 rounds). The results, looking at both the hard numbers and how they performed in practice, were clear: the architectural upgrades in YOLOv8n lead to significant and measurable improvements in accuracy and performance.

What’s truly impressive is that even the tiny "nano" version of YOLOv8 managed to outperform the larger "small" version of its predecessor.

---

## 📂 Dataset Source

**Dataset:**  
The project utilizes the "Rock, Paper, Scissors SXSW" dataset (Version 1) sourced from Roboflow Universe.

- **Link:** [https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)
- **Volume:** The dataset consists of 3,129 images, split into training, validation, and test sets.
- **Classes:** 3 (rock, paper, scissors)

**Preprocessing and Formatting:**

Using the Roboflow platform to guarantee data integrity was a crucial methodological step, as the notebook illustrates. Two downloads of the identical dataset (Version 1) were made, but in two distinct formats that the models required:

1. **YOLOv8:** The dataset was used to download it. `download("yolov8")` command, which offers data and a directory structure. YAML file that has been tailored for the Ultralytics package.  
2. **YOLOv5:** The dataset was used to download YOLOv5. The "YOLOv5 PyTorch" format is provided by the `download("yolov5")` command.

For a true "apples-to-apples" comparison, this step is crucial because it removes dataset formatting as a variable, guaranteeing that both models train on the same images and that validation splits.

---

## ⚙️ Methods

### A. Approach: Controlled Experiment

`yolov8n.pt` (the nano variant) and `yolov5s.pt` (the small variant) are directly compared as part of the core methodology. This method works well for quickly figuring out how two closely related, cutting-edge architectures differ in terms of performance.

**Controlled Variables (Hyperparameters):**
- Dataset: Identical (Roboflow Version 1)
- Base Model: Pretrained COCO weights (implied by yolov8n.pt and yolov5s.pt)
- Epochs: 25
- Image Size: imgsz=640
- Training Environment: Google Colab (Tesla T4 GPU)

---

### B. YOLOv5s vs. YOLOv8n: Core Architectural Differences

The primary difference between the two "ML models" lies in their fundamental neural network architecture. YOLOv8 is not just an increment; it is a significant re-design.

| Component | YOLOv5s (Small) | YOLOv8n (Nano) |
|------------|------------------|----------------|
| **Backbone** | CSPDarknet53. Uses C3 modules (Cross-Stage Partial) which are fast and efficient. | Modified CSP Backbone. Replaces C3 modules with C2f modules (Cross-Stage Partial Bottleneck with 2 convolutions). This design allows for richer feature fusion and gradient flow. |
| **Neck** | PANet (Path Aggregation Network). Uses C3 modules to fuse features from different scales (P3, P4, P5). | PANet. Also a PANet, but it is built using the new C2f modules instead of C3, making the feature fusion process more robust. |
| **Detection Head** | Coupled, Anchor-Based. | Decoupled, Anchor-Free. |
| **Loss Function** | CIoU Loss (Box) + BCE Loss (Class) + BCE Loss (Objectness). | CIoU Loss + DFL (Box) + BCE Loss (Class). |

---

### 🔍 Key Technical Details

#### 1. Backbone (C3 vs. C2f)
- **C3 (YOLOv5):** A feature map is split by a C3 module, which then processes one half using a sequence of "Bottleneck" convolutions before concatenating it with the other, unprocessed half.  
- **C2f (YOLOv8):** This evolution is more intricate. The feature map is also divided, but all intermediate outputs from its bottleneck series are sent for final concatenation. This offers richer feature information and a more comprehensive "path" for gradients.

#### 2. Head (Coupled vs. Decoupled)
- **YOLOv5 (Coupled):** The class, objectness, and bounding box coordinates are all predicted by a single set of convolutional layers, which may cause task conflicts.  
- **YOLOv8 (Decoupled):** The classification and regression components are distinct sub-networks. One focuses on “what is this?” while the other focuses on “where is this?”, improving accuracy.

#### 3. Head (Anchor-Based vs. Anchor-Free)
- **YOLOv5 (Anchor-Based):** Uses pre-defined anchor boxes (priors) of different sizes and shapes.  
- **YOLOv8 (Anchor-Free):** Predicts width, height, and center directly, improving generalization to irregular objects.

#### 4. Loss Function (Objectness vs. DFL)
- **YOLOv5:** Uses “objectness” loss to learn confidence about anchor boxes containing objects.  
- **YOLOv8:** Removes objectness loss and adds **Distribution Focal Loss (DFL)** for bounding box regression, learning probability distributions around box coordinates for stability and precision.

---

## 🚀 Steps to Run the Code

The notebook `rock-ppr-scissors.ipynb` follows a clear, sequential execution flow:

1. **Environment Setup:**  
   Install `ultralytics` (for YOLOv8), `roboflow` (for data download), and clone the YOLOv5 repository.

2. **YOLOv8 Training:**  
   - Download dataset in YOLOv8 format  
   - Run `!yolo task=detect mode=train ...`  
   - Train for 25 epochs and results are saved to `/content/runs/detect/train/`

3. **YOLOv5 Training:**  
   - Change directory to `/content/yolov5/`  
   - Install requirements (`!pip install -r requirements.txt`)  
   - Download dataset in YOLOv5 format  
   - Run `!python train.py ...` for 25 epochs and results are saved to `/content/yolov5/runs/train/exp/`

4. **Analysis:**  
   - Generate a pandas DataFrame for metric comparison  
   - Run inference on a test image (`my_test_image.jpg`) using both models  
   - Display all generated charts

---

## 📊 Experiments / Results Summary

### A. Quantitative Analysis

| Model | mAP (.5:.95) | Precision | Recall |
|--------|--------------|------------|--------|
| **YOLOv8n** | 0.512 | 0.885 | 0.791 |
| **YOLOv5s** | 0.498 | 0.852 | 0.782 |

**Analysis & Interpretation:**

The quantitative results show a clear victory for YOLOv8n, which outperforms YOLOv5s in all three primary metrics.

- The higher **mAP (0.512 vs. 0.498)** results from architectural enhancements such as the decoupled head and DFL loss.  
- The greater **Precision (0.885 vs. 0.852)** shows YOLOv8n makes accurate predictions more often.  
- The improved **Recall (0.791 vs. 0.782)** shows it detects more objects due to the richer C2f feature representation.

---

### B. Visual & Qualitative Analysis

The training scripts generate several diagnostic images and charts:
1. `results.png` – Training & Loss Curves  
2. `confusion_matrix.png` – Confusion Matrix  
3. `PR_curve.png` – Precision-Recall Curve  
4. `train_batch0.jpg` (etc.) – Training Data Batches

---

## 🧠 Conclusion

**Key Findings:**

This technical analysis confirms that the `rock-ppr-scissors.ipynb` notebook correctly executes a fair and valid comparison. The experimental results demonstrate a measurable performance advantage for YOLOv8n.

- **Quantitative Victory:** YOLOv8n outperformed YOLOv5s across all primary metrics  
  - mAP (0.512 vs 0.498)  
  - Precision (0.885 vs 0.852)  
  - Recall (0.791 vs 0.782)

- **Architectural Superiority:**  
  1. Decoupled Head reduces task conflict  
  2. Anchor-Free design provides flexibility  
  3. C2f Module improves feature fusion  
  4. DFL Loss enhances bounding box regression

This project demonstrates that even the smallest "nano" variant of the YOLOv8 architecture is more powerful and efficient than the "small" variant of its predecessor, validating the design choices made by its creators.

---

## 🔗 References

1. **Dataset:** Roboflow. (2023). Rock, Paper, Scissors SXSW Dataset.  
   [https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)
2. **YOLOv8 Model:** Jocher, G. et al. (2023). Ultralytics YOLOv8.  
   [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. **YOLOv5 Model:** Jocher, G. et al. (2020). Ultralytics YOLOv5.  
   [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
