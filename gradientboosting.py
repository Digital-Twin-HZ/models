import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import joblib

# Function to convert Parquet to DataFrame
def convert_df(path):
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
oxygen_df2 = convert_df("../../data/tank1/oxygen_b.parquet")
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

# Synchronizing data
ammonium_df = ammonium_df[ammonium_df.index.isin(water_df.index)]
nitrate_df = nitrate_df[nitrate_df.index.isin(water_df.index)]
phosphate_df = phosphate_df[phosphate_df.index.isin(water_df.index)]
oxygen_df = oxygen_df[oxygen_df.index.isin(water_df.index)]
oxygen_df2 = oxygen_df2[oxygen_df2.index.isin(water_df.index)]
energy_df = energy_df[energy_df.index.isin(water_df.index)]

oxygen_dfs = (oxygen_df + oxygen_df2) / 2
# oxygen_dfs = oxygen_dfs >= 1

# Merging all dataframes
merged_df = pd.concat([phosphate_df, ammonium_df, nitrate_df, energy_df, water_df, oxygen_dfs], axis=1)
merged_df.columns = ['Phosphate', 'Ammonium', 'Nitrate', 'Energy', 'WaterFlow', 'Oxygen']

# Feature Engineering
merged_df['HourOfDay'] = merged_df.index.hour
merged_df['Minute'] = merged_df.index.minute
merged_df['DayOfWeek'] = merged_df.index.dayofweek
merged_df['MonthOfYear'] = merged_df.index.month
merged_df['Season'] = (merged_df.index.month % 12 + 3) // 3

# Splitting data
X = merged_df.drop(['Oxygen'], axis=1)
y = merged_df['Oxygen']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define LSTM Model
class LSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, epochs=50, batch_size=32):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=self.input_shape),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, X, y):
        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for LSTM
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(X_reshaped, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stopping], validation_split=0.1)
        return self

    def predict(self, X):
        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X_reshaped).flatten()

# Train Gradient Boosting model
best_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbose=True)

# Create and Train Voting Regressor
lstm_input_shape = (X_train.shape[1], 1)
voting_regressor = VotingRegressor(
    estimators=[
        ('gb', best_model),
        ('lstm', LSTMRegressor(input_shape=lstm_input_shape, epochs=100, batch_size=32))
    ]
)

voting_regressor.fit(X_train, y_train)

# Predict and evaluate with Voting Regressor
y_pred_ensemble = voting_regressor.predict(X_test)
ensemble_mse = mean_squared_error(y_test, y_pred_ensemble)
ensemble_r2 = r2_score(y_test, y_pred_ensemble)

print(f"Voting Regressor - MSE: {ensemble_mse}, R-squared: {ensemble_r2}")

# Save the models
joblib.dump(best_model, "gradient_boosting_model.joblib")
joblib.dump(voting_regressor, "voting_regressor_model.joblib")

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ensemble}, index=y_test.index)
joblib.dump(results_df, "ensemble_results.joblib")
