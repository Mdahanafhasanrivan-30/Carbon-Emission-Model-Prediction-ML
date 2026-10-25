# Carbon Emission Model Prediction (ML)

This project builds a machine learning pipeline to predict **CO₂ emissions** and classify yearly emission trends (increase/decrease) using country-level historical indicators.

The full implementation is in:
- `CO2emissionModelPrediction_ML.ipynb`

---

## Project Goals

- Predict country-level CO₂ emissions (regression task)
- Classify whether emissions are likely to increase (classification task)
- Compare traditional ML models with deep learning and a hybrid ensemble approach

---

## Dataset

- Source file used in notebook: `CSE427 data.csv`
- Initial shape: **(43,746 rows, 80 columns)**
- After filtering and preprocessing: **(8,598 rows, 73 columns)**
- Final modeling matrix after feature engineering and encoding: **(8,092 rows, 263 features)**

### Preprocessing and Cleaning

- Removed unnecessary columns: `Description`, `iso_code`
- Filtered records for years **>= 1990**
- Dropped columns with more than 50% missing values
- Sorted data by country (`Name`) and year
- Applied missing value handling in stages:
  1. Group-wise interpolation
  2. Group-wise forward/backward fill
  3. Group-wise mean imputation
  4. Global median fallback

---

## Feature Engineering

Created time-aware features:

- `co2_last_year`
- `gdp_last_year`
- `co2_roll_mean` (3-year rolling mean)
- `co2_roll_std` (3-year rolling std)
- `co2_growth`

Targets:

- **Regression target:** `y_reg = log1p(co2)`
- **Classification target:** `y_clf = 1 if co2_growth > 0 else 0`

---

## Feature Selection and Split

- Correlation analysis performed against `y_reg`
- Weakly correlated features removed (`|corr| < 0.05`)
- Potential leakage-related CO₂-derived columns dropped from input space
- Selected core predictors include economic, population, energy, and engineered lag/rolling features
- Country column (`Name`) one-hot encoded

### Train/Test Strategy

Time-based split:

- **Train:** `year <= 2016`
- **Test:** `year > 2016`

Shapes:

- Training: **(6,321, 262)**
- Testing: **(1,771, 262)**

Standard scaling applied before model training.

---

## Handling Class Imbalance

For classification, **SMOTE** is used on training data:

- Before SMOTE: class 1 = 3,823, class 0 = 2,498
- After SMOTE: class 1 = 3,823, class 0 = 3,823

---

## Models Used

### Regression Models

1. Random Forest Regressor
2. XGBoost Regressor
3. BiLSTM Regressor
4. Hybrid Ensemble (weighted):
   - RF weight = 0.40
   - XGB weight = 0.45
   - LSTM weight = 0.15

### Classification Models

1. Random Forest Classifier
2. XGBoost Classifier
3. LSTM-based classifier (thresholding LSTM regression output)
4. Hybrid classifier (thresholding hybrid score)

---

## Results

### Regression Performance

| Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| Random Forest Regressor | 0.999447 | 0.060779 | 0.034608 |
| XGBoost Regressor | 0.999320 | 0.067422 | 0.040537 |
| LSTM Regressor | 0.527499 | 1.777312 | 1.079637 |
| Hybrid RF + XGB + LSTM | 0.988974 | 0.271505 | 0.168079 |

### Classification Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest Classifier | 0.589802 | 0.584746 | 0.746753 | 0.655894 |
| XGBoost Classifier | 0.614164 | 0.625907 | 0.653680 | 0.639492 |
| LSTM Classifier | 0.461756 | 0.485261 | 0.463203 | 0.473976 |
| Hybrid RF + XGB + LSTM | 0.478754 | 0.502268 | 0.479437 | 0.490587 |

### Key Observations

- **Best regression model:** Random Forest (highest R², lowest RMSE/MAE)
- **Best classification accuracy:** XGBoost Classifier
- In this setup, the LSTM and hybrid methods underperform compared to tree-based baselines for classification

---

## Visualizations Included in Notebook

- Correlation heatmap (top correlated features)
- Actual vs predicted emission curves
- Regression metric comparison plots
- Classification metric comparison plots
- Confusion matrices for all classifiers
- Random Forest feature importance chart (top 20)

---

## Installation

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow
```

---

## How to Run

### In Google Colab (as written in notebook)

1. Open `CO2emissionModelPrediction_ML.ipynb`
2. Mount Google Drive
3. Update dataset path if needed (default in notebook):
   - `/content/drive/MyDrive/Cse427/CSE427 data.csv`
4. Run all cells sequentially

### Local Jupyter setup

1. Place dataset in your local environment
2. Update `DATA_PATH` inside the notebook
3. Run all cells in order

---

## Repository Structure

- `CO2emissionModelPrediction_ML.ipynb` — complete end-to-end pipeline
- `README.md` — project documentation

---

## Notes

- This project uses a **time-based train/test split** to respect temporal order.
- Regression target is log-transformed (`log1p`) for stability.
- Classification labels are derived from engineered growth signal.
