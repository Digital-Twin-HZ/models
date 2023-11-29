import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb

def convert_df(path):
    """
    Converts a Parquet file into a Pandas DataFrame.

    Parameters:
    path (str): The path to the Parquet file.

    Returns:
    pandas.DataFrame: The converted DataFrame.
    """
    dframe = pd.read_parquet(path)
    dframe.index = pd.to_datetime(dframe['datumEindeMeting'])
    dframe.drop(columns=['datumEindeMeting', 'datumBeginMeting'], inplace=True)
    dframe = dframe["hstWaarde"].astype(float)
    return dframe[:-1]

# Loading data
ammonium_df = convert_df("../../data/tank1/ammonium.parquet")
nitrate_df = convert_df("../../data/tank1/nitrate.parquet")
phosphate_df = convert_df("../../data/tank1/phosphate.parquet")
oxygen_df = convert_df("../../data/tank1/oxygen_a.parquet")
energy_df = convert_df("../../data/tank1/energy.parquet")
water_df = pd.read_csv("../../data/tank1/water.csv", delimiter=";")

# Preprocessing water data
water_df.index = pd.to_datetime(water_df['DateTime'], format='%d-%m-%Y %H:%M')
water_df['EDE_09902MTW_K100.MTW'] = water_df['EDE_09902MTW_K100.MTW'].str.replace(',', '.').replace('(null)', np.nan, regex=True).astype(float)
water_df['EDE_09902MTW_K100.MTW'] = water_df['EDE_09902MTW_K100.MTW'].interpolate()
water_df = water_df['EDE_09902MTW_K100.MTW']
water_df.index.name = None
water_df = water_df[water_df.index.isin(oxygen_df.index)]
water_df = water_df[~water_df.index.duplicated()]

# Synchronizing data based on datetime index
ammonium_df = ammonium_df[ammonium_df.index.isin(water_df.index)]
nitrate_df = nitrate_df[nitrate_df.index.isin(water_df.index)]
phosphate_df = phosphate_df[phosphate_df.index.isin(water_df.index)]
oxygen_df = oxygen_df[oxygen_df.index.isin(water_df.index)]
energy_df = energy_df[energy_df.index.isin(water_df.index)]

# Merging all dataframes
merged_df = pd.concat([phosphate_df, 
                       ammonium_df, 
                       nitrate_df, 
                       energy_df,
                       water_df,
                       oxygen_df], axis=1)

merged_df.columns = ['Phosphate', 'Ammonium', 'Nitrate', 'Energy', 'WaterFlow', 'Oxygen']

# Feature Engineering: Time-related features
merged_df['HourOfDay'] = merged_df.index.hour
merged_df['Minute'] = merged_df.index.minute
merged_df['DayOfWeek'] = merged_df.index.dayofweek
merged_df['MonthOfYear'] = merged_df.index.month
merged_df['Season'] = (merged_df.index.month % 12 + 3) // 3

# Model preparation
X = merged_df.drop(['Oxygen'], axis=1)
y = merged_df['Oxygen']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define model parameters
lr_params = {'fit_intercept': [True, False]}
dt_params = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}
ridge_params = {'alpha': [10, 100, 200]}
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
knn_params = {'n_neighbors': [3, 5, 10]}
svr_params = {'C': [1, 10], 'gamma': ['scale', 'auto']}
xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}

# Create parameter grids
lr_grid = list(ParameterGrid(lr_params))
dt_grid = list(ParameterGrid(dt_params))
ridge_grid = list(ParameterGrid(ridge_params))
rf_grid = list(ParameterGrid(rf_params))
gb_grid = list(ParameterGrid(gb_params))
knn_grid = list(ParameterGrid(knn_params))
svr_grid = list(ParameterGrid(svr_params))
xgb_grid = list(ParameterGrid(xgb_params))

ensemble_predictions = []
weights = []

'''
Linear Regression (LR):
Strengths: Simple and interpretable. Performs well when the relationship between features and the target variable is linear.
Usefulness: Provides a baseline for performance and helps in understanding linear relationships in the data.

Decision Tree Regressor (DT):
Strengths: Handles non-linear relationships well. Easy to interpret and understand.
Usefulness: Captures complex patterns in the data, which linear models might miss.

Ridge Regression:
Strengths: Similar to linear regression but includes regularization to prevent overfitting. Works well with multicollinearity.
Usefulness: Improves on linear regression by addressing overfitting, especially useful when dealing with high-dimensional data.

Random Forest Regressor:
Strengths: An ensemble of decision trees, typically more accurate than a single decision tree. Handles non-linear data well.
Usefulness: Brings robustness and reduces overfitting risk, capturing complex interactions between features.

Support Vector Regression (SVR):
Strengths: Effective in high-dimensional spaces, robust to outliers.
Usefulness: Provides good generalization capabilities, especially in complex but sparse datasets.

Gradient Boosting Regressor:
Strengths: Builds trees one at a time, where each new tree helps to correct errors made by previous ones. Often very effective.
Usefulness: Offers high accuracy and can handle different types of data (linear, non-linear).

K-Nearest Neighbors Regressor (KNN):
Strengths: Simple and effective, particularly useful in scenarios where the decision boundary is very irregular.
Usefulness: Can capture complex patterns by considering the 'closeness' of data points.

XGBoost Regressor:
Strengths: An optimized gradient boosting library. Fast, efficient, and often outperforms other models.
Usefulness: Highly scalable and can handle a variety of data types, making it a go-to model for many competitions and real-world applications.

By combining these diverse models, the ensemble benefits from their collective strengths. This approach helps in:
Reducing Overfitting: Different models have different tendencies to overfit; an ensemble can average out these tendencies.
Handling Different Data Types: Some models work better with linear relationships, others with non-linear. An ensemble can capture a wider range of patterns.
Improving Predictive Performance: Often, an ensemble outperforms any single model, especially on complex tasks.
Robustness: The ensemble is less likely to be thrown off by peculiarities of the data that might affect a single model disproportionately.
In summary, each model contributes its unique strengths to the ensemble, leading to a more robust, accurate, and generalizable predictive model.
'''

# Iterative model building and ensemble creation
for params in lr_grid + dt_grid + ridge_grid + rf_grid + gb_grid + knn_grid + svr_grid + xgb_grid:
    if params in lr_grid:
        model = LinearRegression(**params)
    elif params in dt_grid:
        model = DecisionTreeRegressor(**params, random_state=42)
    elif params in ridge_grid:
        model = Ridge(**params)
    elif params in rf_grid:
        model = RandomForestRegressor(**params, random_state=42)
    elif params in gb_grid:
        model = GradientBoostingRegressor(**params, random_state=42)
    elif params in knn_grid:
        model = KNeighborsRegressor(**params)
    elif params in svr_grid:
        model = SVR(**params)
    elif params in xgb_grid:
        model = xgb.XGBRegressor(**params, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ensemble_predictions.append(y_pred)

    # The idea is that models with lower MSE (better performance) should have a higher weight in the final ensemble prediction. By taking the inverse, a smaller MSE (which indicates a better model) results in a larger weight.
    mse = mean_squared_error(y_test, y_pred)
    weights.append(1 / mse)

    weighted_preds = np.average(ensemble_predictions, axis=0, weights=weights)
    ensemble_mse = mean_squared_error(y_test, weighted_preds)
    ensemble_r2 = r2_score(y_test, weighted_preds)

    print(f"Iteration {len(ensemble_predictions)} - Ensemble MSE: {ensemble_mse}, R-squared: {ensemble_r2}")

# Final ensemble prediction
final_ensemble_pred = np.average(ensemble_predictions, axis=0, weights=weights)

# Model Evaluation and Saving Results
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': final_ensemble_pred}, index=y_test.index)
joblib.dump(results_df, "results_ensemble.joblib")