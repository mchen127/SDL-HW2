import argparse
import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import src.config as config
from src.dataset import JenaClimateDataset, engineer_features
from src.model import LSTMForecast


def _parse_window_args():
    """Parse CLI overrides for window-related hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Train the LSTM forecaster with custom window settings."
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=config.INPUT_WIDTH,
        help="Number of past timesteps provided to the model (default: %(default)s)",
    )
    parser.add_argument(
        "--label-width",
        type=int,
        default=config.LABEL_WIDTH,
        help="How many future target steps to predict (default: %(default)s)",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=config.SHIFT,
        help="Gap between the last input point and the start of the label window (default: %(default)s)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=None,
        help="Dimensionality of the model output. Defaults to label width when omitted.",
    )

    args = parser.parse_args()
    args.output_size = args.output_size if args.output_size is not None else args.label_width
    return args


def _validate_window_params(label_width: int, output_size: int) -> None:
    if label_width != output_size:
        raise ValueError(
            "label_width and output_size must match so the loss compares tensors of the same shape."
        )


def train_model(
    input_width: int = config.INPUT_WIDTH,
    label_width: int = config.LABEL_WIDTH,
    shift: int = config.SHIFT,
    output_size: int = config.OUTPUT_SIZE,
):
    """
    Main function to train the LSTM model.
    """
    _validate_window_params(label_width, output_size)
    # --- 1. Load and Preprocess Data ---
    print("Loading pre-split data...")
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    val_df = pd.read_csv(config.VAL_CSV_PATH)

    print("Engineering features...")
    train_df_fe = engineer_features(train_df)
    val_df_fe = engineer_features(val_df)

    # --- 2. Scale Data ---
    print("Scaling data...")
    scaler = StandardScaler()
    # Fit scaler ONLY on the training data
    scaler.fit(train_df_fe)

    # Save the scaler for use in evaluation
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    scaler_filepath = config.scaler_path(input_width, label_width, shift)
    joblib.dump(scaler, scaler_filepath)
    print(f"Scaler saved to {scaler_filepath}")

    # Transform all datasets
    train_df_scaled = pd.DataFrame(
        scaler.transform(train_df_fe), index=train_df_fe.index, columns=train_df_fe.columns
    )
    val_df_scaled = pd.DataFrame(
        scaler.transform(val_df_fe), index=val_df_fe.index, columns=val_df_fe.columns
    )

    # --- 3. Create Datasets and DataLoaders ---
    print("Creating datasets and dataloaders...")
    train_dataset = JenaClimateDataset(
        train_df_scaled, input_width, label_width, shift
    )
    val_dataset = JenaClimateDataset(
        val_df_scaled, input_width, label_width, shift
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 4. Initialize Model, Loss, and Optimizer ---
    print(f"Using device: {config.DEVICE}")
    model = LSTMForecast(
        config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, output_size
    ).to(config.DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- 5. Training Loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_filepath = config.model_path(input_width, label_width, shift)
            torch.save(model.state_dict(), model_filepath)
            print(f"Model saved to {model_filepath}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")


if __name__ == "__main__":
    # Before running, make sure you have generated the split CSVs
    # by running `python src/preprocess.py`
    if not all([os.path.exists(p) for p in [config.TRAIN_CSV_PATH, config.VAL_CSV_PATH]]):
        print("Processed data not found. Please run 'python src/preprocess.py' first.")
    else:
        args = _parse_window_args()
        train_model(
            input_width=args.input_width,
            label_width=args.label_width,
            shift=args.shift,
            output_size=args.output_size,
        )
