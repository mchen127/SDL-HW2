# Project Plan: Jena Climate Forecasting

## 1. Project Goal

The primary objective is to build, train, and evaluate a Long Short-Term Memory (LSTM) neural network using PyTorch to forecast air temperature. The model will use 24 hours of historical data from 14 climate features to predict the temperature 1 hour into the future.

## 2. Data

*   **Source**: Jena Climate dataset (`jena_climate_2009_2016.csv`).
*   **Features**: 14 features including pressure, humidity, wind velocity, etc.
*   **Target Variable**: Air Temperature `T (degC)`.
*   **Granularity**: Raw data is recorded every 10 minutes.

## 3. Project Structure and Rationale

The project is structured to promote modularity, reusability, and a clear workflow. This separation of concerns is a key design decision.

### `src/preprocess.py`
This script is responsible for the initial, one-time data preparation.
- **Load**: Loads the raw `jena_climate_2009_2016.csv`.
- **Clean**: Replaces erroneous `-9999.0` values in wind velocity columns with `0.0`.
- **Downsample**: Resamples the 10-minute data to 1-hour intervals by taking the mean. This reduces noise and computational load.
- **Forward-fill**: After resampling, use forward-fill (`ffill`) to propagate the last valid observation forward. This is a standard, causal method to handle missing values in time-series data without introducing data leakage, and it ensures the sequence remains unbroken for the LSTM model.
- **Split**: Splits the cleaned, downsampled data chronologically into `train.csv`, `val.csv`, and `test.csv` (70-20-10 split).
- **Output**: Saves the three new CSV files to the `problem2/data/` directory.

### `src/dataset.py`
This module focuses on preparing the data for the model.
- **`engineer_features`**: A function that takes a DataFrame and creates cyclical features. It converts wind direction (`wd (deg)`) and the timestamp into `sin`/`cos` components to make their cyclical nature explicit to the model.
- **`JenaClimateDataset`**: A PyTorch `Dataset` class that takes a feature-engineered and scaled DataFrame and creates sliding windows of (input, label) pairs, where the input is 24 hours of data and the label is the temperature 1 hour ahead.

### `src/model.py`
Defines the neural network architecture.
- **`LSTMForecast`**: A PyTorch `nn.Module` class that defines a simple but effective LSTM network with a final linear layer to produce the temperature forecast.

### `src/config.py`
A centralized configuration file.
- Contains all hyperparameters, file paths, data split ratios, and training parameters. This allows for easy tuning without modifying the core logic.

### `src/main.py`
The main script for orchestrating the training process.
- Loads `train.csv` and `val.csv`.
- Calls `engineer_features` to create cyclical features.
- **Crucially, it fits a `StandardScaler` on the training data *only*** to prevent data leakage.
- Saves the fitted scaler to `problem2/results/` for use during evaluation.
- Transforms the training and validation sets using the fitted scaler.
- Creates PyTorch `DataLoader`s for batching.
- Implements the training loop with early stopping to prevent overfitting.
- Saves the best performing model to `problem2/results/`.

### `src/eval.py`
A dedicated script for final model evaluation on the unseen test set.
- Loads `test.csv`.
- Applies the same feature engineering.
- **Loads the saved `scaler.pkl`** from the results directory and transforms the test data. This ensures consistent scaling.
- Loads the saved `best_model.pth`.
- Runs the model on the test data and reports the final performance metrics (e.g., Mean Absolute Error).

## 4. Workflow

The end-to-end process is executed with three main commands:

1.  **Prepare Data (Run once)**:
    ```bash
    python -m src.preprocess
    ```

2.  **Train Model**:
    ```bash
    python -m src.main
    ```

3.  **Evaluate Model**:
    ```bash
    python -m src.eval
    ```

## 5. Key Design Decisions

- **Modularity**: Separating the one-time data preparation (`preprocess.py`) from the repeatable training workflow (`main.py`) makes the codebase cleaner and more maintainable.
- **Data Leakage Prevention**: The `StandardScaler` is fitted *only* on the training data and then used to transform all three data splits. This is critical for obtaining a reliable measure of the model's performance.
- **Reproducibility**: By saving the exact scaler and the best model, the evaluation process is completely reproducible and independent of the training script.
- **Cyclical Features**: Transforming time and wind direction into `sin`/`cos` components is a standard best practice for time-series and cyclical data, as it helps the model learn the patterns more effectively.
- **Configuration Management**: Using a `config.py` file makes it easy to experiment with different settings and track hyperparameters.