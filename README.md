---


### *Calorie Estimation Project — End-to-End Pipeline from Image → Segmentation → Classification → Mass → Calories*

---

## **📌 Project Overview**

The **Calorie Estimation Project** builds a complete, modular pipeline that estimates calorie content from food images. It integrates multiple computer vision tasks and structured datasets to convert raw images into interpretable calorie predictions.

The system uses **FoodSeg103**, **Nutrition5k**, and **USDA FoodData Central** and combines segmentation, classification, depth/mass estimation, and ML calorie prediction into one unified workflow.

This repository contains all notebooks (converted to Markdown for GitHub readability) that implement each stage of the pipeline.

---

## **📁 Project Structure**

```
/Calorie_Estimation_Project
│
├── Markdown Notebooks/   # .md versions of all project notebooks (readable on GitHub)
├── notebooks/            # Original .ipynb notebooks (not rendered on GitHub)
├── data/                 # Datasets (ignored in GitHub)
├── cache/                # Cached intermediate features (ignored)
├── checkpoints/          # Model weights (ignored)
├── logs/                 
├── maskrcnn_outputs/
├── Outputs/
└── README.md
```

The **Markdown Notebooks** folder contains the readable version of every notebook and forms the basis for this README.

---

## **🧠 Problem Statement**

Food logging is inaccurate when done manually. The goal of this project is to estimate:

* **Food category**
* **Portion size (mass/volume)**
* **Calories**

from a **single RGB image** (or optionally, multi-view samples).

The workflow builds a reproducible benchmark pipeline that approximates academic calorie-estimation systems using publicly available datasets.

---

## **🔄 End-to-End Pipeline Summary**

### **1. Data Collection & Management (Notebook 01)**

Datasets loaded and verified:

* **FoodSeg103** → food images + pixel-wise segmentation masks
* **Nutrition5k** → paired RGB + depth images + dish-level metadata
* **USDA FoodData Central** → reference nutrients for calorie mapping

A manifest system ensures reproducibility and organizes dataset paths and metadata across notebooks.

---

### **2. Data Preprocessing (Notebook 02)**

Key preprocessing includes:

* Loading and validating FoodSeg103 image–mask pairs
* Decoding Nutrition5k RGB and depth images from byte format
* Extracting relevant USDA nutrients (energy, protein, carbs, fat)
* Saving processed subsets to `/cache` and `/processed`

This prepares unified samples for segmentation, feature extraction, and modeling.

---

### **3. Exploratory Modeling & Visualization (Notebook 03)**

Performed detailed EDA:

* Visualization of sample FoodSeg103 images + masks
* Visualization of RGB + depth from Nutrition5k
* Statistical exploration of USDA nutrient distributions
* Correlation studies for calories vs nutrients
* Scatter plots for calories vs protein/fat/carbs
* Construction of processed subsets for modeling

This stage reveals the relationships necessary for downstream prediction.

---

### **4. Image Segmentation (Notebook 04)**

Built segmentation pipeline using **Mask R-CNN** and/or pretrained segmentation utilities.

Primary outputs:

* Extracted masks per food instance
* Bounding box generation
* Extraction of per-item segmented crops
* Area-based features (e.g., pixel area)
* Visualization of segmentation overlays (in notebook)

Output mask data is stored in `/maskrcnn_outputs`.

---

### **5. Food Classification (Notebook 05)**

A food-category classifier was developed using:

* **FoodSeg103’s 103 food classes**
* Segment-level crops from segmentation stage
* ConvNext/TIMM backbone
* Train/val splits ensuring class coverage

Includes:

* Caching 3,000-image subset with complete class coverage
* Label extraction from category_id.txt
* Masked crop extraction and standard transforms
* Model training with AMP, timm backbones, and ConvNext-L
* Tracking metrics across epochs

The notebook outputs classification predictions and saves features for calorie estimation.

---

### **6. Mass Prediction (Notebook 06)**

This notebook uses Nutrition5k’s structured metadata to learn mapping from image-derived geometric/visual features to **mass in grams**.

Core components:

* Loading geometry features (MiDaS volume, mean height, pixel area)
* Loading visual features (CLIP/ViT principal components)
* Integration with Nutrition5k ground truth mass
* Feature engineering:

  * log, sqrt, square, cubic transformations
  * mass × visual interactions
  * mass × geometry interactions
* Standardization using `StandardScaler`
* 10-fold training of:

  * **LightGBM**
  * **XGBoost**
  * **CatBoost**
* Stacking ensemble for best accuracy
* Saving predictions to `/Outputs`

Mass predictions feed directly into the final calorie model.

---

### **7. Calorie Prediction (Notebook 07)**

The final stage estimates **calories per dish**.

Using:

* Predicted mass (or true mass when available)
* Geometry features
* Visual features
* Nutrition5k ground truth calories

Steps included:

* Merging geometry + visual + metadata
* Cleaning infinite/missing values
* Building unified training and test splits
* Feature engineering
* 10-fold ensemble modeling (LGBM/XGB/CatBoost)
* Calibration using **Isotonic Regression**
* Final calorie predictions saved to `/Outputs`

This completes the pipeline from image → mass → calories.

---

## **🔥 Key Contributions**

* **Fully reproducible pipeline across 7 notebooks**
* **Complete integration of three major datasets** (FoodSeg103, Nutrition5k, USDA)
* **Instance segmentation pipeline** with high-quality mask extraction
* **103-class food classifier** using advanced ConvNext/TIMM backbones
* **Mass estimation ensemble** with strong predictive accuracy
* **Calorie prediction using engineered visual + geometry + mass features**
* **Robust caching layer** for fast development and recomputation
* **Dataset manifests and clear folder hierarchy** for long-term maintainability

---

## **🚀 How to Use This Repository**

1. Clone repository
2. Add datasets under `data/raw` using Notebook 01 or manually
3. Run notebooks in order 01 → 07
4. Use Markdown versions on GitHub for quick review
5. Outputs appear in:

   * `/cache`
   * `/maskrcnn_outputs`
   * `/Outputs`
   * `/checkpoints`

---

## **👤 Author**

**Venkatesh Katragadda**
Calorie Estimation AI — End-to-End Benchmark Study

---

