# Geospatial Tree Detection & Vegetation Analysis (NDVI)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-000000?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX Runtime-Execution-005BA1?style=flat-square)](https://onnxruntime.ai/)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-41CD52?style=flat-square)](https://pypi.org/project/PyQt5/)
[![Rasterio](https://img.shields.io/badge/Rasterio-GIS Data-2B5C8F?style=flat-square)](https://rasterio.readthedocs.io/)

An end-to-end computer vision and remote sensing project for detecting, counting, and analyzing forest canopy health from multi-band satellite images (GeoTIFF) using a fine-tuned YOLOv8 model deployed on an interactive Desktop GUI.

---

## Overview

Monitoring forest density and vegetation health is critical for environmental tracking and resource management. This application combines **deep learning object detection** with **geospatial remote sensing indices**:

1. **Tree Detection & Automated Counting:** Utilizes a custom-trained YOLOv8 model deployed via ONNX Runtime using image tiling and Non-Maximum Suppression (NMS) for full-resolution satellite coverage.
2. **Vegetation Indexing (NDVI):** Computes normalized difference vegetation indices directly from multi-band satellite imagery to assess canopy health in both grayscale and Viridis color heatmaps.

---

## Model Fine-Tuning & Performance

The detection pipeline was trained by fine-tuning YOLOv8 on a dedicated forest canopy dataset using online data augmentations (Mosaic, Mixup, scale/shear transformations, and CLAHE).

### Evaluation Metrics

| Metric | Score / Result |
| :--- | :--- |
| **Precision (P)** | **82.31%** |
| **Recall (R)** | **82.76%** |
| **F1 Score** | **82.53%** |
| **mAP@50** | **87.76%** |
| **mAP@50-95** | **57.66%** |

---

## Desktop Viewer Application (PyQt5)

A dedicated desktop GUI built with `PyQt5` provides researchers and operators with full control over GeoTIFF visualization and automated tree processing.

### Key GUI Features
* **Multi-Spectral Display Modes:**
  * **RGB View:** Renders normalized 16-bit GeoTIFF bands to 8-bit visual space using percentile stretching.
  * **NDVI Grayscale:** Computes NIR/Red band ratios $\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$ to highlight bare soil versus vegetation.
  * **NDVI Viridis Heatmap:** Maps vegetation density to a color scale ranging from purple (low vegetation) to vibrant yellow (healthy dense canopy).
* **Automated Tree Counting:**
  * Slices large satellite scenes into uniform $640 \times 640$ tiles with edge padding.
  * Executes fast CPU inference using `onnxruntime`.
  * Reconstructs global bounding boxes and displays total tree counts with visual bounding indicators.
* **Interactive Navigation:** Supports full mouse wheel zooming (Ctrl + Scroll) and canvas drag-and-pan controls.

---

## Technical Stack & Dependencies

* **Deep Learning Framework:** PyTorch, Ultralytics YOLOv8, Roboflow
* **Deployment Format:** ONNX Runtime (`CPUExecutionProvider`)
* **Geospatial & Image Libraries:** Rasterio, OpenCV (`cv2`), NumPy
* **User Interface:** PyQt5
