# Insurance Claim Fraud Detection

Machine learning project to predict whether a car insurance claim is fraudulent using the `insurance_claims.csv` dataset.

## Overview

This project builds an end-to-end classification pipeline for fraud detection in vehicle insurance claims. The workflow includes data loading, exploratory data analysis, cleaning, missing value treatment, categorical encoding, feature selection, model comparison, hyperparameter tuning, and a final deep learning benchmark.

The main goal is to classify new claims as:

- **Fraudulent**
- **Non-fraudulent**

## Project Goal

The objective is to support early fraud detection in an insurance context by training a model that can generalize to new claims and identify suspicious cases with high predictive performance.

## Dataset

The dataset used in this project is:

- `insurance_claims.csv`

It contains structured information about policyholders, vehicles, incidents, claim details, and the target variable `fraud_reported`.

## Repository Contents

- `insurance_claims.csv` — source dataset
- `fraud_team.py` — Python script with the full pipeline
- `fraud_team.ipynb` — Jupyter notebook with the exploratory analysis and experimentation
- `proyecto_final.pdf` — final project presentation
- `README.md` — project documentation

## Methodology

### 1. Data loading and initial exploration

- Dataset loaded with `pandas`
- Target variable converted from `Y/N` to binary format
- Histograms generated to inspect distributions and class imbalance

### 2. Data cleaning

- Duplicate rows removed
- Constant / uninformative columns removed
- Columns with little analytical value discarded
- Missing values studied with `missingno`
- Categorical variables encoded with `OrdinalEncoder`

### 3. Missing value treatment

Several imputation strategies were evaluated using cross-validation and execution time comparison:

- SimpleImputer (`mean`, `median`, `most_frequent`, `constant`)
- `KNNImputer`
- `IterativeImputer`

After comparison, **KNN Imputer with 3 neighbors** was selected as the best option for this dataset.

### 4. Feature analysis and extraction

The project studies feature relevance using several techniques:

- Correlation matrix
- RFE / RFECV
- PCA
- Truncated SVD

### 5. Feature selection

Categorical and numerical variables were analyzed separately using:

- **Chi-square test**
- **Mutual information**
- **ANOVA (f_classif)**
- **Permutation importance**

This helped identify the least relevant variables and improve model performance by removing them.

### 6. Model comparison

The following classification models were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- KNN-based permutation importance analysis

### 7. Hyperparameter tuning

Since **XGBoost** produced the best results, it was selected as the final classical ML model.

A **3-round Grid Search refinement process** was applied to tune parameters such as:

- `max_depth`
- `learning_rate`
- `gamma`
- `reg_lambda`
- `scale_pos_weight`
- `subsample`
- `colsample_bytree`
- `n_estimators`

### 8. Deep learning benchmark

A neural network was also trained using **Keras** to compare performance with the classical machine learning models.

## Main Results

The best result was obtained with **XGBoost**, reaching approximately **89% accuracy on the test set** after feature selection and hyperparameter tuning.

The neural network achieved around **81% accuracy**, but it did not outperform XGBoost on this dataset.

### Final XGBoost parameters

```python
{
    'colsample_bytree': 0.5,
    'gamma': 0.2,
    'learning_rate': 0.03,
    'max_depth': 3,
    'n_estimators': 150,
    'reg_lambda': 15,
    'scale_pos_weight': 3,
    'subsample': 0.8
}
```

## Why XGBoost was selected

XGBoost was chosen as the final model because it delivered the best balance between predictive performance and robustness after comparing multiple approaches. It outperformed logistic regression, decision trees, random forests, and the neural network benchmark.

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- keras / tensorflow
- missingno
- Jupyter Notebook

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook fraud_team.ipynb
```

### 4. Or run the script directly

```bash
python fraud_team.py
```

## Key Learnings

- Complete machine learning workflow from raw data to final prediction
- Practical handling of missing values and categorical features
- Feature selection using statistical methods and model-based importance
- Comparison of several classification algorithms
- Hyperparameter tuning with Grid Search
- Benchmarking classical ML against a neural network

## Conclusions

This project demonstrates a full fraud detection pipeline for car insurance claims. After analyzing the dataset, cleaning the data, selecting relevant variables, and comparing multiple models, XGBoost was the most effective solution for this problem.

The project also shows that deep learning is not always the best choice for tabular data when the dataset size is limited and tree-based models can capture the underlying patterns more efficiently.

## Author

**Luis Alex Farfan**

---

