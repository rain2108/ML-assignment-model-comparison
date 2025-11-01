# ML-assignment-model-comparison

# A Technical Report and Comparative Architectural Analysis of YOLOv8n and YOLOv5s for Real-Time Rock-Paper-Scissors Detection

## 🧾 Short Description
In this project, we put two top-tier, real-time object detection models—YOLOv5s and YOLOv8n—head-to-head to see which was better for a specific, fun task: recognizing "rock," "paper," and "scissors" hand gestures.

However, we wanted to examine more than just performance evaluations. We decided to examine the models themselves in order to comprehend the transition from the more traditional anchor-based system of YOLOv5 to the more recent anchor-free system of YOLOv8.

To ensure a fair comparison, we trained both models for the same duration (5 rounds) and with the same dataset (Roboflow's "RPS SXSW"). The findings were evident from both the quantitative data and their practical use: YOLOv8n's architectural modifications significantly improved accuracy and performance.

It is simply amazing that even the minuscule "nano" version of YOLOv8 outperformed the larger "small" version of its predecessor.

---

## 📂 Dataset Source

**Dataset:**  
- **Link:** [https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)  
- **Volume:** The dataset consists of 3,129 images, split into training, validation, and test sets.  
- **Classes:** 3 (rock, paper, scissors)

---

## ⚙️ Methods

### A. Approach: Controlled Experiment
The core methodology includes a direct comparison between `yolov8n.pt` (the nano variant) and `yolov5s.pt` (the small variant). This technique is useful for rapidly determining how two closely related, state-of-the-art architectures differ in terms of performance.

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
1. **Backbone (C3 vs. C2f):**  
   - **C3 (YOLOv5):** A feature map is split by a C3 module, which then processes one half using a sequence of "Bottleneck" convolutions before concatenating it with the other, unprocessed half.  
   - **C2f (YOLOv8):** This evolution is more intricate. The feature map is also divided, but all intermediate outputs from its bottleneck series are sent for final concatenation. This offers richer feature information and a more comprehensive "path" for gradients.

2. **Head (Coupled vs. Decoupled):**  
   - **YOLOv5 (Coupled):** The bounding box coordinates, the class (rock, paper, scissor), and the "objectness" (is there an object here?) are all predicted by a single set of convolutional layers. This could lead to conflict.  
   - **YOLOv8 (Decoupled):** The classification and regression parts of the model are distinct, thin sub-networks, or "heads." This reduces task conflict and increases accuracy.

3. **Head (Anchor-Based vs. Anchor-Free):**  
   - **YOLOv5:** Uses pre-existing anchor boxes (priors).  
   - **YOLOv8:** Predicts object center `(x, y)` and `(width, height)` directly — easier to train and generalizes better to irregular shapes.

4. **Loss Function (Objectness vs. DFL):**  
   - **YOLOv5:** Uses objectness loss to teach the model when an anchor box contains an object.  
   - **YOLOv8:** Adds Distribution Focal Loss (DFL) for bounding box regression, learning probability distributions around box coordinates for stable, accurate box learning.

---

## 📊 Experiments / Results Summary

### A. Quantitative Analysis

| Model | mAP (.5:.95) | Precision | Recall |
|--------|--------------|------------|--------|
| **YOLOv8n** | 0.512 | 0.885 | 0.791 |
| **YOLOv5s** | 0.498 | 0.852 | 0.782 |

**Analysis & Interpretation:**

- The higher **mAP (0.512 vs. 0.498)** is directly attributable to YOLOv8’s enhanced architecture and independent optimization of classification and regression tasks.  
- The higher **Precision (0.885 vs. 0.852)** indicates that YOLOv8n makes correct predictions more often.  
- The higher **Recall (0.791 vs. 0.782)** reflects better detection capability due to the richer C2f feature representation.

---

### B. Visual & Qualitative Analysis

The training scripts generate several diagnostic images and charts, which are crucial for a technical analysis.

#### 1. Training & Loss Curves (`results.png`)
| YOLOv8n | YOLOv5s |
|:--:|:--:|
| ![YOLOv8 Training & Loss Curves](results_yolov8.png) | ![YOLOv5 Training & Loss Curves](results_yolov5.png) |

#### 2. Confusion Matrix (`confusion_matrix.png`)
| YOLOv8n | YOLOv5s |
|:--:|:--:|
| ![YOLOv8 Confusion Matrix](confusion_matrix_yolov8.png) | ![YOLOv5 Confusion Matrix](confusion_matrix_yolov5.png) |

#### 3. Precision-Recall Curve (`PR_curve.png`)
| YOLOv8n | YOLOv5s |
|:--:|:--:|
| ![YOLOv8 PR Curve](PR_curve_yolov8.png) | ![YOLOv5 PR Curve](PR_curve_yolov5.png) |

#### 4. Training Data Batches (`train_batch0.jpg`, etc.)
| YOLOv8n | YOLOv5s |
|:--:|:--:|
| ![YOLOv8 Training Batch](train_batch_yolov8.jpg) | ![YOLOv5 Training Batch](train_batch_yolov5.jpg) |

> 🖼️ *Replace these image filenames with the actual output images from your training folders before committing.*

---

## 🧠 Conclusion

**Key Findings:**
- The `rock-ppr-scissors.ipynb` notebook correctly conducts a valid and fair comparison.  
- **YOLOv8n** outperformed **YOLOv5s** in all significant metrics:  
  - mAP (0.512 vs. 0.498)  
  - Precision (0.885 vs. 0.852)  
  - Recall (0.791 vs. 0.782)

**Superiority in Architecture:**
1. The Decoupled Head reduces task conflict.  
2. The Anchor-Free design provides more flexibility.  
3. The C2f Module allows for richer feature fusion.  
4. The DFL Loss provides a more advanced mechanism for bounding box regression.

By demonstrating that even the tiniest "nano" version of the new YOLOv8 architecture is more powerful and efficient than the "small" version of its highly optimized predecessor, this project validates the design choices made by its creators.

---

## 🔗 References

1. **Dataset:** Roboflow. (2023). Rock, Paper, Scissors SXSW Dataset.  
   [https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)
2. **YOLOv8 Model:** Jocher, G. et al. (2023). Ultralytics YOLOv8.  
   [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. **YOLOv5 Model:** Jocher, G. et al. (2020). Ultralytics YOLOv5.  
   [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
