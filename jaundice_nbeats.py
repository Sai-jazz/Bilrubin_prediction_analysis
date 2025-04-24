import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
import os
import tensorflow as tf

# Constants
MODEL_PATH = "bilirubin_nbeats_model.h5"
SCALER_PATH = "scaler.pkl"
EPOCHS = 150  # Slightly increased for N-BEATS
BATCH_SIZE = 4  # Reduced for small dataset
FORECAST_STEPS = 5  # Number of future steps to predict

# Hardcoded baseline data
baseline_days = np.array([1, 3, 4, 6, 8, 13, 23])
baseline_bilirubin = np.array([5.2, 8.7, 8.8, 8.5, 6.3, 4.1, 2.3])

# Modified sequence creation for N-BEATS
def create_sequences(days, bilirubin, n_steps=3):
    X, y = [], []
    for i in range(len(bilirubin) - n_steps - FORECAST_STEPS + 1):
        X.append(bilirubin[i:i + n_steps])
        y.append(bilirubin[i + n_steps:i + n_steps + FORECAST_STEPS])
    return np.array(X), np.array(y)

# N-BEATS Model builder
def build_nbeats_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Generic block function
    def create_block(input_layer, block_num):
        # Shared layers
        fc1 = Dense(64, activation='relu', name=f'block_{block_num}_fc1')(input_layer)
        fc2 = Dense(64, activation='relu', name=f'block_{block_num}_fc2')(fc1)
        
        # Forecast and backcast branches
        theta_f = Dense(64, activation='relu', name=f'block_{block_num}_theta_f')(fc2)
        forecast = Dense(FORECAST_STEPS, name=f'block_{block_num}_forecast')(theta_f)
        
        theta_b = Dense(64, activation='relu', name=f'block_{block_num}_theta_b')(fc2)
        backcast = Dense(input_shape[0], name=f'block_{block_num}_backcast')(theta_b)
        
        return backcast, forecast
    
    # Stack 3 blocks (reduced for small dataset)
    residuals = inputs
    forecasts = []
    
    for i in range(3):
        backcast, forecast = create_block(residuals, i)
        residuals = Subtract(name=f'subtract_block_{i}')([residuals, backcast])
        forecasts.append(forecast)
    
    # Sum all forecasts
    total_forecast = Add(name='final_forecast')(forecasts)
    
    model = Model(inputs=inputs, outputs=total_forecast)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                  loss='mse',
                  metrics=['mae'])
    return model

# Model trainer
def train_model(X_train, y_train, X_test, y_test):
    model = build_nbeats_model((X_train.shape[1], 1))
    checkpoint = ModelCheckpoint(MODEL_PATH, 
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    return model, history

# Prediction (modified for N-BEATS direct forecasting)
def predict_future(model, scaler, last_sequence):
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    predictions = model.predict(last_sequence_scaled.reshape(1, -1, 1), verbose=0)[0]
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    return predictions

# CSV training loader (modified for N-BEATS)
def load_training_data_from_csv(file_path, n_steps=3):
    try:
        df = pd.read_csv(file_path)
        required_cols = {'PatientID', 'Day', 'Bilirubin'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain the columns: {required_cols}")

        scaler = MinMaxScaler()
        all_sequences_X = []
        all_sequences_y = []

        for patient_id, patient_df in df.groupby('PatientID'):
            patient_df = patient_df.sort_values('Day')
            bilirubin = patient_df['Bilirubin'].values

            if len(bilirubin) < n_steps + FORECAST_STEPS:
                continue

            bilirubin_scaled = scaler.fit_transform(bilirubin.reshape(-1, 1)).flatten()
            X, y = create_sequences(np.arange(len(bilirubin_scaled)), bilirubin_scaled, n_steps)
            all_sequences_X.append(X)
            all_sequences_y.append(y)

        if not all_sequences_X:
            raise ValueError("No patient has enough data to generate sequences.")

        X = np.vstack(all_sequences_X)
        y = np.vstack(all_sequences_y)

        return X, y, scaler

    except Exception as e:
        raise RuntimeError(f"Error processing CSV for training: {e}")

# Plotting function (updated for multi-step prediction)
def plot_comparison(patient_data, predictions, baseline_data):
    plt.figure(figsize=(12, 6))

    # Baseline data
    plt.plot(baseline_data['days'], baseline_data['bilirubin'], 
             'b-', marker='o', markersize=8, linewidth=2,
             label=f"Baseline (Final: {baseline_data['bilirubin'][-1]:.1f} mg/dl)")

    # Patient data
    plt.plot(patient_data['days'], patient_data['bilirubin'],
             'r-', marker='s', markersize=8, linewidth=2,
             label=f"Patient (Current: {patient_data['bilirubin'][-1]:.1f} mg/dl)")

    # Predictions
    future_days = np.linspace(patient_data['days'][-1], 
                             patient_data['days'][-1] + len(predictions),
                             len(predictions) + 1)[1:]
    plt.plot(future_days, predictions, 'g--', marker='^', markersize=8, linewidth=2,
            label=f"Predicted (Final: {predictions[-1]:.1f} mg/dl)")

    plt.title("Bilirubin Level Comparison", fontsize=14, pad=20)
    plt.xlabel("Days Since Birth", fontsize=12)
    plt.ylabel("Bilirubin (mg/dl)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='upper right')

    max_day = max(np.max(baseline_data['days']), np.max(patient_data['days']), np.max(future_days))
    plt.xlim(0, max_day * 1.1)

    min_bili = min(np.min(baseline_data['bilirubin']), np.min(patient_data['bilirubin']), np.min(predictions))
    max_bili = max(np.max(baseline_data['bilirubin']), np.max(patient_data['bilirubin']), np.max(predictions))
    plt.ylim(min_bili * 0.9, max_bili * 1.1)

    plt.tight_layout()
    plt.show()

# Main routine
def main():
    print("\n" + "="*50)
    print("Bilirubin Level Monitoring System (N-BEATS Version)")
    print("="*50)

    csv_path = input("Enter the training CSV file path (or press Enter to use baseline): ").strip()

    if os.path.exists(MODEL_PATH) and csv_path == "":
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("\nLoaded pre-trained model")
    else:
        if not os.path.exists(csv_path):
            print("\nUsing baseline data for training...")
            # Create synthetic data from baseline if no CSV provided
            days = baseline_days
            bilirubin = baseline_bilirubin
            X, y = create_sequences(days, bilirubin)
            scaler = MinMaxScaler().fit(bilirubin.reshape(-1, 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        else:
            print("\nTraining model from CSV data...")
            X, y, scaler = load_training_data_from_csv(csv_path)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model, _ = train_model(X_train, y_train, X_test, y_test)
        joblib.dump(scaler, SCALER_PATH)
        print("Model trained and saved.")

    while True:
        try:
            days_patient, bilirubin_patient = input_manual_data()
            if len(bilirubin_patient) < 3:
                print("Need at least 3 measurements to make predictions")
                continue
                
            last_sequence = bilirubin_patient[-3:]  # Last 3 readings
            predictions = predict_future(model, scaler, last_sequence)

            baseline_data = {'days': baseline_days, 'bilirubin': baseline_bilirubin}
            plot_comparison(
                patient_data={'days': days_patient, 'bilirubin': bilirubin_patient},
                predictions=predictions,
                baseline_data=baseline_data
            )

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please check your inputs and try again")

        if input("\nAnalyze another patient? (yes/no): ").strip().lower() != 'yes':
            print("\nThank you for using the Bilirubin Monitoring System!")
            break

if __name__ == "__main__":
    main()