# Kaggle House Prices: Advanced Regression Techniques

**Author:** Mohamed Amine Tarik  
**Status:** Completed  
**Goal:** Predict final house prices using advanced regression techniques and feature engineering.

---

##  Project Overview
This project is a solution for the famous [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The objective is to predict the sales price of houses in Ames, Iowa, based on 79 explanatory variables describing (almost) every aspect of residential homes.

The solution uses a professional Data Science workflow, featuring robust data cleaning, domain-specific feature engineering, and an ensemble of linear and tree-based models.

##  Project Structure

```text
kaggle-house-prices/
├── data/
│   ├── train.csv           # Raw training data
│   ├── test.csv            # Raw test data
│   ├── processed/          # Cleaned and transformed data (X_train, X_test, etc.)
│   └── submission.csv      # Final predictions for Kaggle
├── models/
│   ├── lasso_tuned.pkl     # Serialized Lasso model (Best Alpha)
│   └── xgboost.pkl         # Serialized XGBoost model
├── notebooks/
│   ├── 01-eda-and-setup.ipynb  # Exploratory Analysis, Cleaning, & Feature Engineering
│   └── 02-modeling.ipynb       # Cross-Validation, Hyperparameter Tuning, & Ensembling
└── README.md

 Methodology
1. Data Preprocessing
Target Transformation: Applied log1p transform to SalePrice to correct high skewness and satisfy the RMSLE evaluation metric.

Outlier Removal: Identified and removed bivariate outliers in GrLivArea vs SalePrice.

Missing Value Imputation:

Categorical (e.g., PoolQC): Imputed with "None" (indicating absence).

Numerical (e.g., GarageArea): Imputed with 0.

LotFrontage: Imputed using the median of the specific Neighborhood.

2. Feature Engineering
Creation: Created aggregated features like TotalSF (Total Square Footage), TotalBath, and GarageAge.

Transformation: Applied log1p transform to all skewed numerical features (skew > 0.75).

Encoding: Used One-Hot Encoding for categorical variables to handle non-ordinal relationships (e.g., Neighborhood).

3. Modeling & Strategy
Validation: Used 5-Fold Cross-Validation (KFold) to ensure model stability.

Model 1: Lasso Regression (L1):

Used for automatic feature selection (sparse coefficients).

Hyperparameters tuned via GridSearchCV.

Model 2: XGBoost:

Used to capture non-linear relationships and complex interactions.

Ensemble:

Blended predictions (50% Lasso + 50% XGBoost) to reduce variance and improve generalization.

 Results
Evaluation Metric: Root Mean Squared Logarithmic Error (RMSLE).

CV Performance: ~0.1112 (Lasso) / ~0.1169 (XGBoost).

 How to Run
Clone the repository.

Install dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn.

Run notebooks/01-eda-and-setup.ipynb to generate processed data.

Run notebooks/02-modeling.ipynb to train models and generate submission.csv.