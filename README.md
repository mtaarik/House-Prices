# Kaggle House Prices: Advanced Regression Techniques

Predicting the sale price of houses in Ames, Iowa, from 79 explanatory variables
([Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)).
The work focuses on careful preprocessing, domain-driven feature engineering and a blend of a
linear model and a tree-based model, all validated with 5-fold cross-validation.

## Results

Metric: RMSE on `log1p(SalePrice)`, which is the competition's RMSLE. 5-fold cross-validation on the
training set.

| Model | CV RMSE (log) | Std |
|---|---|---|
| Ridge | 0.1189 | 0.0095 |
| **Lasso (tuned)** | **0.1112** | 0.0076 |
| XGBoost | 0.1169 | 0.0053 |
| Final submission | 50% Lasso + 50% XGBoost blend | |

Lasso kept 120 of the 347 engineered features, which also gives a readable view of the price
drivers.

## Methodology

### 1. Preprocessing
- **Target:** `log1p` transform of `SalePrice` to correct its strong right skew and match the metric.
- **Outliers:** removal of the bivariate outliers in `GrLivArea` vs `SalePrice`.
- **Missing values:**
  - categorical features where missing means "absent" (e.g. `PoolQC`) are filled with `"None"`;
  - numerical features where missing means zero (e.g. `GarageArea`) are filled with `0`;
  - `LotFrontage` is filled with the median of its `Neighborhood`.

### 2. Feature engineering
- New aggregate features: `TotalSF` (total square footage), `TotalBath`, `GarageAge`.
- `log1p` transform of every numerical feature with skewness above 0.75.
- One-hot encoding of the categorical features (e.g. `Neighborhood`).

### 3. Modeling
- **Validation:** 5-fold cross-validation (`KFold`).
- **Lasso (L1):** automatic feature selection through sparse coefficients, `alpha` tuned with
  `GridSearchCV`.
- **XGBoost:** captures non-linear effects and interactions.
- **Blend:** the average of both predictions reduces variance.

## Repository structure

```text
Notebooks/
  01-eda-and-setup.ipynb   exploration, cleaning and feature engineering
  02-modeling.ipynb        cross-validation, tuning and blending
data/
  data_description.txt     description of the 79 variables
```

## Run it

1. Download `train.csv` and `test.csv` from the
   [competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   into `data/`.
2. Install the dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`.
3. Run `Notebooks/01-eda-and-setup.ipynb` to build the processed data, then
   `Notebooks/02-modeling.ipynb` to train the models and write `submission.csv`.
