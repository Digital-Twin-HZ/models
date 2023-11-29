# GradientBoosting + Deep Learning VotingRegressor Documentation: Predicting Dissolved Oxygen Levels in Water Treatment Plant

## Overview
This document serves as the documentation for a predictive model designed to estimate dissolved oxygen levels in a water treatment plant. The model leverages various measurements from the plant to make its predictions.

## Data Sources and Preprocessing

### Data Loading
- **Sources**: Data is sourced from Parquet and CSV files, representing different measurements from a water treatment plant.
- **Measurements**:
  - Ammonium levels (`ammonium.parquet`)
  - Nitrate levels (`nitrate.parquet`)
  - Phosphate levels (`phosphate.parquet`)
  - Oxygen levels (`oxygen_a.parquet` and `oxygen_b.parquet`)
  - Energy consumption (`energy.parquet`)
  - Water flow data (`water.csv`)

### Data Preparation
- **Function `convert_df`**: A custom function used to convert Parquet files into Pandas DataFrames. It sets the index to the datetime column, removes unnecessary columns, and converts the target variable to a float type.
- **Preprocessing Water Data**: The `water.csv` file is processed to format datetime, handle null values, and interpolate missing data.

### Synchronizing DataFrames
- Ensures all data frames are aligned in time, focusing on the same timestamps across all measurements.

## Feature Engineering
- Adds time-related features like hour of the day, day of the week, and month of the year to the dataset. Also calculates the season based on the month.

## Model Development

### LSTM Model
- **Class `LSTMRegressor`**: A custom regressor using Long Short-Term Memory (LSTM) layers, suitable for time-series data.
- **Training**: The model is trained with early stopping to prevent overfitting.

### Gradient Boosting Regressor
- A Gradient Boosting model is trained for comparison and ensemble purposes.

### Ensemble Model: Voting Regressor
- **Combination**: The model combines predictions from the Gradient Boosting model and LSTM model.
- **Purpose**: This approach aims to leverage strengths from both models to improve overall prediction accuracy.

## LSTMRegressor Class Details

### Purpose
- The `LSTMRegressor` class is designed to handle sequential, time-series data effectively. Its architecture is tailored for learning from temporal patterns in data, which is essential for predicting dissolved oxygen levels that vary over time.

### Architecture
- **Input Layer**: Takes reshaped input suitable for LSTM (3D array).
- **LSTM Layers**: Two LSTM layers with 50 units each. The first LSTM layer returns sequences, providing a full sequence to the second LSTM layer.
- **Output Layer**: A Dense layer with a single unit for regression output.

### Training Process
- **Reshaping Data**: Input data is reshaped into a 3D array to fit the LSTM's requirements.
- **Compilation**: The model is compiled with the mean squared error loss function and the Adam optimizer.
- **Early Stopping**: Implemented to halt training when the validation loss stops improving, preventing overfitting.

### Integration
- This LSTM model is integrated into the Voting Regressor, where it collaborates with a Gradient Boosting model, balancing the predictive capabilities of both.

## Model Evaluation
- The models are evaluated using Mean Squared Error (MSE) and R-squared metrics, ensuring a robust assessment of their performance.

## Saving the Models
- The final models and their predictions are saved using `joblib` for future use and analysis.

## Usage
- This model can be used for real-time prediction of dissolved oxygen levels in water treatment plants, aiding in efficient and effective plant management.