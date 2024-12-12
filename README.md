# IndustryGuard™
<img src="industryguard.gif" width=800>


## Introduction

This project develops an object detection system using YOLOv8 to identify people and Personal Protective Equipment (PPE) in images. It involves converting annotations from PascalVOC to YOLOv8 format, training separate YOLOv8 models for person and PPE detection, and building an inference pipeline with visualization.

## Repository Structure

```plaintext
├── dataset/
│   ├── images
│   └── labels
│   └── classes.txt
├── training/
│   ├── person_det_model_training.ipynb
│   └── PPE_detection_model_training.ipynb
├── weights/
│   ├── person_detection_model.pt
│   └── ppe_detection_model.pt
└── inference.py
└── pascalVOC_to_yolo.py
└── README.md
```

- **Conversion:** Python script (`pascalVOC_to_yolo.py`) converts PascalVOC annotations to YOLOv8 format.
- **Model Training:** Trained YOLOv8 models for detecting people and PPE (hard-hat, gloves, mask, boots, vest, PPE-suit).
- **Inference:** Python script (`inference.py`) performs detection and visualizes results with bounding boxes and confidence scores.

## Steps

### 1. Data Preprocessing
- Resized images to 640x640.
  
 <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/2.png" width=200>_____<img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/3.png" width=350>
- Balanced data by combining the `mask` class with `PPE suit`.
 <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/4.png" width=400>

### 2. PascalVOC to YOLOv8 Conversion
- Converted PascalVOC XML annotations to YOLOv8 text format with normalized coordinates and class IDs.

### 3. YOLOv8 Model Training

- **Person Detection Model:**
  - Trained on person-only data.
  - Epochs: 150, Batch size: 8.
   
     <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/5.png" width=375>
      <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/6.png" width=550>

- **PPE Detection Model:**
  - Trained on cropped images with PPE labels.
  - Epochs: 200, Batch size: 8.
     
     <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/8.png" width=375>
      <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/9.png" width=550>

### 4. Inference Script

- Detects people and PPE in images.
- Crops images to detected persons for PPE detection.
- Visualizes results with OpenCV.

## Results
-
     <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/10.png" width=375>
      <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/11.png" width=550>
  <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/12.png" width=550>

## Conclusion

The project successfully developed a YOLOv8-based detection system for people and PPE. Future work may include real-time deployment in surveillance or safety monitoring applications.


---
