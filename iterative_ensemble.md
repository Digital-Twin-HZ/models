# Model Documentation: Ensemble Approach for Predicting Dissolved Oxygen Levels

## Overview
This document serves as documentation for an ensemble-based predictive model designed to estimate dissolved oxygen levels in a water treatment plant, using various measurements from the plant.

## Data Sources and Preprocessing

### Data Loading
- **Sources**: Data is sourced from Parquet and CSV files, representing different measurements from a water treatment plant.
- **Measurements**:
  - Ammonium levels (`ammonium.parquet`)
  - Nitrate levels (`nitrate.parquet`)
  - Phosphate levels (`phosphate.parquet`)
  - Oxygen levels (`oxygen_a.parquet`)
  - Energy consumption (`energy.parquet`)
  - Water flow data (`water.csv`)

### Data Preparation
- **Function `convert_df`**: Converts Parquet files into Pandas DataFrames. Sets the index to datetime and processes the target variable.
- **Preprocessing Water Data**: Formats datetime, handles null values, and interpolates missing data in `water.csv`.

### Synchronizing DataFrames
- Aligns all data frames based on datetime index to ensure consistency across measurements.

## Feature Engineering
- Adds time-related features to the dataset: hour of the day, minute, day of the week, month of the year, and season.

## Model Development

### Ensemble Model Approach
- **Models Used**: Linear Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors, Support Vector Regression, and XGBoost.
- **Parameters**: Defined various hyperparameters for each model to explore during training.
- **Ensemble Strategy**: Uses a weighted average approach where each model's prediction is weighted inversely proportional to its mean squared error.

### Iterative Model Building
- Trains each model iteratively over the defined parameter grids.
- Calculates ensemble predictions and weights after each iteration, based on model performance.
- Evaluates the ensemble model using MSE and R-squared metrics.

### Strengths and Usefulness of Each Model
- **Linear Regression**: Provides a baseline for performance and helps understand linear relationships.
- **Decision Tree**: Captures complex patterns in the data, useful for non-linear relationships.
- **Ridge Regression**: Addresses overfitting, particularly effective with high-dimensional data.
- **Random Forest**: More robust than a single decision tree, reduces overfitting risk.
- **Support Vector Regression**: Effective in high-dimensional spaces and robust to outliers.
- **Gradient Boosting**: Often highly accurate, handling different data types effectively.
- **K-Nearest Neighbors**: Useful in scenarios with irregular decision boundaries.
- **XGBoost**: Fast, efficient, and often outperforms other models in a variety of data types.

## Ensemble Model Benefits
- **Reducing Overfitting**: Balances out individual model tendencies to overfit.
- **Handling Different Data Types**: Captures a wide range of patterns, both linear and non-linear.
- **Improving Predictive Performance**: Often outperforms any single model.
- **Robustness**: Less susceptible to data peculiarities affecting a single model.

## Model Evaluation and Saving Results
- Final ensemble prediction is evaluated and results are saved using `joblib`.

## Usage
- This ensemble model can be used for real-time prediction of dissolved oxygen levels in water treatment plants, aiding in efficient and effective plant management.
