# House Price Prediction

Predicting house prices using linear regression, based on features like area, 
bedrooms, bathrooms, and amenities.

## Dataset
Housing Prices Dataset (Kaggle) — 545 rows, features including area, bedrooms, 
bathrooms, stories, and binary amenities (mainroad, guestroom, basement, AC, parking).

## Approach
1. EDA — checked distributions, missing values, correlation heatmap
2. Categorical encoding via one-hot encoding (`pd.get_dummies`)
3. Outlier detection using IQR method — found 15 outliers, all on the expensive end
4. Capped outliers at the upper IQR bound (winsorizing) instead of dropping rows
5. Trained Linear Regression and Ridge Regression, compared performance

## Results
| Model | RMSE | R² |
|---|---|---|
| Linear Regression (raw) | 13,24,507 | 0.653 |
| Linear Regression (capped) | 11,29,536 | 0.684 |
| Ridge Regression (capped) | 11,30,440 | 0.684 |

Capping outliers improved RMSE by ~₹1.95L and R² by ~3 points. Ridge regularization 
had negligible additional effect, suggesting the model wasn't overfitting on large 
coefficients — the outliers were the main source of error, not model complexity.

## What I'd try next
- Polynomial features to capture non-linear relationships
- Lasso regression for feature selection
- Try a tree-based model (Random Forest) for comparison
