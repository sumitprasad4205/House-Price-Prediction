
## 🏠 House Price Prediction Model

This project evaluates and compares different machine learning regression models to predict housing prices. By leveraging ensemble learning and hyperparameter tuning, the project aims to minimize prediction error and maximize variance explanation ($R^2$).

## 🚀 Model Performance Summary

A comparative analysis was conducted across three model configurations: **Random Forest**, **XGBoost**, and a **Hyperparameter-Tuned XGBoost**.

| Model | $R^2$ Score | RMSE | MSE |
| --- | --- | --- | --- |
| **Tuned XGBoost** | **0.842** | **45,525.51** | **2,072,580,861.50** |
| XGBoost Regressor | 0.829 | 47,350.72 | 2,242,090,716.72 |
| Random Forest Regressor | 0.817 | 48,941.70 | 2,395,290,032.47 |

### Key Takeaways

* **Best Performer:** The **Tuned XGBoost** model achieved the highest accuracy, explaining **84.2%** of the variance in house prices ($R^2 = 0.842$) with the lowest error rates.
* **Error Reduction:** Hyperparameter tuning successfully reduced the Root Mean Squared Error (RMSE) by approximately **$1,825** compared to the baseline XGBoost model.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** `pandas`, `scikit-learn`, `xgboost`
