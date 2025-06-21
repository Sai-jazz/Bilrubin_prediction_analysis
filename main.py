#main code...

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import os
import tensorflow as tf

# Constants
MODEL_PATH = "bilirubin_lstm_model.h5"
SCALER_PATH = "scaler.pkl"
EPOCHS = 100
BATCH_SIZE = 8

# Hardcoded baseline data
baseline_days = np.array([1, 3, 4, 6, 8, 13, 23])
baseline_bilirubin = np.array([5.2, 8.7, 8.8, 8.5, 6.3, 4.1, 2.3])

# Sequence creation
def create_sequences(days, bilirubin, n_steps=3):
    X, y = [], []
    for i in range(len(bilirubin) - n_steps):
        X.append(bilirubin[i:i + n_steps])
        y.append(bilirubin[i + n_steps])
    return np.array(X), np.array(y)

# Model builder
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse',
                  metrics=['mae'])
    return model

# Model trainer
def train_model(X_train, y_train, X_test, y_test):
    model = build_lstm_model((X_train.shape[1], 1))
    checkpoint = ModelCheckpoint(MODEL_PATH, 
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min')
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=1
    )
    return model, history

# Prediction
def predict_future(model, scaler, last_sequence, future_steps=5):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(future_steps):
        next_pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)[0, 0]
        predictions.append(next_pred)
        current_seq = np.append(current_seq[1:], next_pred)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions

# CSV training loader
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

            if len(bilirubin) < n_steps + 1:
                continue

            bilirubin_scaled = scaler.fit_transform(bilirubin.reshape(-1, 1)).flatten()
            X, y = create_sequences(np.arange(len(bilirubin_scaled)), bilirubin_scaled, n_steps)
            all_sequences_X.append(X)
            all_sequences_y.append(y)

        if not all_sequences_X:
            raise ValueError("No patient has enough data to generate sequences.")

        X = np.vstack(all_sequences_X)
        y = np.concatenate(all_sequences_y)

        return X, y, scaler

    except Exception as e:
        raise RuntimeError(f"Error processing CSV for training: {e}")

# Manual input for predictions
def input_manual_data():
    days = []
    bilirubin = []
    print("\nEnter bilirubin data (day and value). Type 'done' when finished (minimum 3 entries required).")

    while True:
        day_input = input("Day (e.g., 1, 3, 5...): ").strip()

        if day_input.lower() == 'done':
            if len(days) >= 3:
                break
            print("Please enter at least 3 data points before typing 'done'.")
            continue

        try:
            day = float(day_input)
            if day <= 0:
                print("Day must be positive.")
                continue

            value = float(input(f"Bilirubin level (mg/dl) for Day {day}: ").strip())
            if value < 0:
                print("Bilirubin level cannot be negative.")
                continue

            days.append(day)
            bilirubin.append(value)
            print(f"Added Day {day}: {value} mg/dl")

        except ValueError:
            print("Invalid input. Please enter numbers only.")

    sorted_idx = np.argsort(days)
    return np.array(days)[sorted_idx], np.array(bilirubin)[sorted_idx]

# Plotting function
def plot_comparison(patient_data, predictions, baseline_data):
    plt.figure(figsize=(12, 6))

    plt.plot(baseline_data['days'], baseline_data['bilirubin'], 
             'b-', marker='o', markersize=8, linewidth=2,
             label=f"Baseline (Final: {baseline_data['bilirubin'][-1]:.1f} mg/dl)")

    plt.plot(patient_data['days'], patient_data['bilirubin'],
             'r-', marker='s', markersize=8, linewidth=2,
             label=f"Patient (Current: {patient_data['bilirubin'][-1]:.1f} mg/dl)")

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
    print("Bilirubin Level Monitoring System")
    print("="*50)

    csv_path = input("Enter the training CSV file path: ").strip()

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("\nLoaded pre-trained model")
    else:
        print("\nTraining model from CSV data...")
        X, y, scaler = load_training_data_from_csv(csv_path)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        model, _ = train_model(X_train, y_train, X_test, y_test)
        joblib.dump(scaler, SCALER_PATH)
        print("Model trained and saved.")

    while True:
        try:
            days_patient, bilirubin_patient = input_manual_data()
            bilirubin_scaled = scaler.transform(bilirubin_patient.reshape(-1, 1)).flatten()
            X_patient, _ = create_sequences(days_patient, bilirubin_scaled)
            last_seq = X_patient[-1]
            future_preds = predict_future(model, scaler, last_seq)

            baseline_data = {'days': baseline_days, 'bilirubin': baseline_bilirubin}

            plot_comparison(
                patient_data={'days': days_patient, 'bilirubin': bilirubin_patient},
                predictions=future_preds,
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