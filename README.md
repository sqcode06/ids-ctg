# Predicting Fetal Health from Cardiotocography (CTG) Data

**Author**: _Oleksandr Mazur_  
**Course**: Introduction to Data Science (IDS 2025)  
**Project ID**: A6 | KAGGLE–FETAL–HEALTH

---

## 1. Motivation and goal

Cardiotocography (CTG) is used in obstetrics to monitor fetal heart rate and uterine contractions. Visual interpretation of CTG traces is subjective and depends on clinician experience. The goal of this project is to use a labelled CTG dataset from Kaggle to:

- build and compare machine-learning models that predict fetal health (normal / suspect / pathological), and
- design evaluation and decision rules that focus on clinically relevant errors (especially underestimating suspect or pathological cases).

The poster summarises the main results; this repository contains the code used to produce them.

---

## 2. Dataset

**Name:** Fetal Health Classification (Kaggle)  
**Link:** https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

Each row corresponds to one CTG measurement with numeric features describing heart-rate variability, accelerations/decelerations, uterine contractions and histogram descriptors, plus a `fetal_health` label:

- 1 – normal
- 2 – suspect
- 3 – pathological

The notebook expects the dataset file to be available as:

```text
fetal_health.csv
```

in the repository root. If you store it somewhere else, update the ``DATA_PATH`` variable near the top of ``main.ipynb``.

---

## 3. Repository structure

```
.
├─ README.md              # This file
├─ main.ipynb             # Full analysis notebook
├─ fetal_health.csv       # Kaggle dataset
├─ figures/               # PNGs exported for the poster
│   ├─ corr_matrix.png
│   ├─ cm_HGB_tuned.png
│   ├─ cm_HGB_tuned_rep.png
│   ├─ r_p_suspect_zoom.png
│   ├─ r_p_pathological.png
│   └─ cm_HGB_hierarchical.png
└─ A6_report.pdf          # The HW10 project report
```

Only ``main.ipynb`` is needed to reproduce the analysis; the images in ``figures/`` are exported versions used on the poster.

---

## 4. Code overview

All analysis lives in the Jupyter notebook main.ipynb, which is organised into the following sections:

1. **Predicting Fetal Health from Cardiotocography (CTG) Data**  
Project title, link to the dataset and a bullet-point overview of the steps.<br><br>
2. **Imports & basic configuration**  
Loads Python libraries (NumPy, pandas, matplotlib, SciPy, scikit-learn, XGBoost), sets a random seed and defines DATA_PATH and TARGET_COLUMN.<br><br>
3. **Load data**  
Reads fetal_health.csv into a pandas DataFrame.<br><br>
4. **Basic inspection / class distribution / histograms**  
Prints dataset shape and dtypes, summarises numerical columns, plots the class distribution and histograms for selected features.<br><br>
5. **Correlation matrix for numerical features**  
Computes the Pearson correlation matrix between all numeric features and visualises it as a heatmap. This figure is used as Fig. 1 on the poster.<br><br>
6. **Clustering of features & selections of representative features**  
Uses hierarchical clustering on the absolute correlation matrix to group highly correlated features, then picks one representative feature per cluster. Later, a gradient-boosting model is trained both on all features and on these representatives for comparison.<br><br>
7. **Label encoding**  
Encodes the fetal health classes 1/2/3 as 0/1/2 with LabelEncoder and defines helper functions for mapping back to readable class names.<br><br>
8. **Train/test split**  
Performs an 80/20 stratified train–test split on the encoded labels.<br><br>
9. **Evaluation helpers**  
Functions for training a model, printing accuracy/macro-F1/recall by class, and drawing (normalised) confusion matrices.<br><br>
10. **Baseline models**  
Trains and evaluates several baselines: a dummy classifier, logistic regression, decision tree and random forest. These provide a performance reference.<br><br>
11. **Advanced models – HistGradientBoosting & XGBoost**  
Tunes a HistGradientBoosting classifier and an XGBoost classifier using grid search with stratified cross-validation. The tuned HGB model is selected as final_model because it slightly outperforms XGBoost and is more stable.<br><br>
12. **Threshold sweeps for pathological and suspect classes**  
For each of the minority classes (2 = suspect, 3 = pathological), treats it as a binary “class vs others” problem and sweeps a decision threshold on its predicted probability. This produces recall/precision/accuracy curves used in Fig. 3 on the poster.<br><br>
13. **Hierarchical decision rule and evaluation**  
Implements and evaluates a hierarchical rule:<br><br>
    - if $P(pathological) \geq t_3$ → pathological
    - else if $P(suspect) \geq t_2$ → suspect
    - else → normal<br><br>

    The section prints overall accuracy, macro F1, class-wise recalls, counts of under-/over-estimation errors, and plots the final confusion matrix (Fig. 3c).<br><br>
14. **Choosing thresholds for evaluation** 
Sets concrete values for $t_3$ and $t_2$ based on the threshold sweeps and re-runs the hierarchical evaluation to obtain the final results.

---

## 5. How to run the notebook and reproduce the analysis

1. **Install dependencies (e.g. in a virtual environment):**
    ```shell
    ~$ pip install numpy pandas matplotlib scipy scikit-learn xgboost pillow
    ```
    Any recent Python 3.10+ version should work.<br><br>
2. **Ensure the dataset** ``fetal_health.csv`` is in the repository root (or adjust ``DATA_PATH`` in
``main.ipynb``).<br><br>
3. **Open the notebook:**  

   For example:
   ```shell
   ~$ jupyter lab
   ```
   And open ``main.ipynb``.<br><br>
4. **Run all cells from top to bottom.** This will:
    - reproduce the EDA plots (class distribution, correlation matrix),
    - train and evaluate baseline and advanced models,
    - run threshold sweeps for suspect and pathological classes,
    - evaluate the hierarchical decision rule and generate the confusion matrices.<br><br>
    Due to randomness in train–test splitting and cross-validation, the exact numeric values may differ slightly from the poster, but the overall behaviour (which class is hardest, shape of the curves, etc.) should be the same.

---

## 6. **Extra code vs poster content**  
The repository may contain code that is not directly illustrated on the poster (e.g. random forest
baseline, clustering details, different threshold values). This is intentional and shows the full
extent of the experimentation that was done during the project, even if only a subset of the
results could fit into the final A0 poster.