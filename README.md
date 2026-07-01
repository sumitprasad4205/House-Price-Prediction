🏡 California Housing Price Prediction
This project predicts median house prices using the California Housing dataset. It uses a machine learning pipeline with preprocessing and a Random Forest Regressor to ensure consistent training and inference.

📊 Model Performance
RMSE: 49,101.03
MSE: 2,410,911,309.34
R² Score: 0.8177
The model explains ~81% of the variance in housing prices, showing good predictive performance.

⚙️ Project Workflow


Load dataset (housing.csv)

Create income-based stratified split (income_cat)

Split data into training and test sets

Preprocess data:

Numerical: median imputation + scaling

Categorical: One-Hot Encoding (ocean_proximity)

Build preprocessing pipeline using ColumnTransformer

Train RandomForestRegressor model

Evaluate model using RMSE, MSE, and R² score

Save trained model and pipeline using joblib

Run inference on new data and export predictions to output.csv

🧠 Tech Stack:

Python
Pandas, NumPy
Scikit-learn
Joblib

📈 Future Improvements

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
Try XGBoost / LightGBM for better accuracy
Feature engineering (location clustering)
Deploy using Flask / FastAPI

