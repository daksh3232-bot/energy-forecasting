
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

def main():
    print("Fetching dataset...", flush=True)
    try:
        ds = fetch_ucirepo(id=235)
        X_raw = ds.data.features
        y_raw = ds.data.targets
        df = pd.concat([X_raw, y_raw], axis=1)
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    # Preprocessing
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df[['Global_active_power']].fillna(df['Global_active_power'].mean())

    # Sliding Window
    def create_sliding_window(df, window_size=30, forecast_horizon=120):
        X, y = [], []
        data = df['Global_active_power'].values
        for i in range(window_size, len(data) - forecast_horizon):
            X.append(data[i-window_size:i])
            y.append(data[i+1:i+forecast_horizon+1])
        return np.array(X), np.array(y)

    window_size = 30
    forecast_horizon = 120
    X, y = create_sliding_window(df, window_size, forecast_horizon)

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Limit training size to 5000 
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    # Small test set for speed in this script if needed, but let's use full test for accuracy unless too slow
    # X_test = X_test[:1000] 
    # y_test = y_test[:1000]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Data prepared. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}", flush=True)

    # 1. SVM
    print("Training SVM (SVR)...", flush=True)
    # Reducing C or increasing epsilon can speed it up, but let's stick to valid params
    svm_model = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1), n_jobs=-1)
    svm_model.fit(X_train_scaled, y_train)
    print("SVM Trained. Predicting...", flush=True)
    y_pred_svm = svm_model.predict(X_test_scaled)

    mae_svm = mean_absolute_error(y_test, y_pred_svm)
    mse_svm = mean_squared_error(y_test, y_pred_svm)
    r2_svm = r2_score(y_test, y_pred_svm)
    print(f"METRICS_SVM: MAE={mae_svm:.4f}, MSE={mse_svm:.4f}, R2={r2_svm:.4f}", flush=True)

    # 2. Random Forest
    print("Training Random Forest...", flush=True)
    rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1), n_jobs=-1) # Reduced estimators for speed check
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest Trained. Predicting...", flush=True)
    y_pred_rf = rf_model.predict(X_test_scaled)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"METRICS_RF: MAE={mae_rf:.4f}, MSE={mse_rf:.4f}, R2={r2_rf:.4f}", flush=True)

if __name__ == "__main__":
    main()
