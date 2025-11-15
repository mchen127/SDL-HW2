import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

import src.config as config
from src.dataset import JenaClimateDataset, engineer_features
from src.model import LSTMForecast


def evaluate_model():
    """
    Loads the test data and a trained model to evaluate its performance.
    """
    # --- 1. Load Test Data and Artifacts ---
    if not all([os.path.exists(p) for p in [config.TEST_CSV_PATH, config.SCALER_PATH, config.MODEL_PATH]]):
        print("Ensure test data, scaler, and model exist. Run preprocess.py and main.py first.")
        return

    print("Loading test data and artifacts...")
    test_df = pd.read_csv(config.TEST_CSV_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    # --- 2. Process Test Data ---
    print("Processing test data...")
    test_df_fe = engineer_features(test_df)

    # Ensure columns are in the same order as when the scaler was fitted
    # This is crucial for correct scaling
    test_df_scaled = pd.DataFrame(
        scaler.transform(test_df_fe), index=test_df_fe.index, columns=test_df_fe.columns
    )

    # --- 3. Create Test DataLoader ---
    test_dataset = JenaClimateDataset(
        test_df_scaled, config.INPUT_WIDTH, config.LABEL_WIDTH, config.SHIFT
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 4. Load Model ---
    print("Loading model...")
    model = LSTMForecast(
        config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.OUTPUT_SIZE
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    # --- 5. Make Predictions ---
    print("Making predictions on the test set...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)

    # --- 6. Inverse Transform for Interpretation ---
    # We need the mean and scale of the target feature ONLY
    target_col_index = test_df_fe.columns.get_loc(config.TARGET_FEATURE)
    temp_mean = scaler.mean_[target_col_index]
    temp_scale = scaler.scale_[target_col_index]

    # Inverse transform predictions and labels
    preds_unscaled = (preds_np * temp_scale) + temp_mean
    labels_unscaled = (labels_np * temp_scale) + temp_mean

    # --- 7. Calculate and Report Metrics ---    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(preds_unscaled - labels_unscaled))
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((preds_unscaled - labels_unscaled)**2))
    # MAPE (Mean Absolute Percentage Error) - with epsilon to avoid division by zero
    mape = np.mean(np.abs((labels_unscaled - preds_unscaled) / (labels_unscaled + 1e-6))) * 100

    print(f"\n--- Test Set Evaluation Metrics ---")
    print(f"\nTest Set Mean Absolute Error (MAE): {mae:.4f} °C")
    print(f"Test Set Root Mean Squared Error (RMSE): {rmse:.4f} °C")
    print(f"Test Set Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # --- 8. Visualize Results ---
    print("Visualizing predictions...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 6))
    # Plot a slice of the test set to keep the plot readable
    plot_slice = slice(100, 200)
    plt.plot(labels_unscaled[plot_slice], label="Actual Temperature", marker='o', linestyle='-')
    plt.plot(preds_unscaled[plot_slice], label="Predicted Temperature", marker='x', linestyle='--')
    plt.title("Temperature Forecast vs. Actual (Test Set Slice)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(config.RESULTS_DIR, "test_predictions.png")
    plt.savefig(plot_path)
    print(f"Prediction plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    evaluate_model()