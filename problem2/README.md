# Jena Climate Forecasting with PyTorch LSTM

## Overview

This project aims to predict future air temperature using a Long Short-Term Memory (LSTM) network. The model is built with PyTorch and trained on the Jena Climate dataset, which contains 14 different weather-related features recorded every 10 minutes from 2009 to 2016.

The core task is to forecast the temperature 1 hour into the future based on the previous 24 hours of observations.

## Project Structure

The project is organized into a modular structure to separate concerns like data preprocessing, model definition, and training.

```
problem2/
├── data/
│   └── jena_climate_2009_2016.csv  # Raw dataset
├── docs/
│   └── plan.md                     # Detailed project plan
├── results/                        # Output directory for model and scaler
├── src/
│   ├── config.py                   # Central configuration for all parameters
│   ├── dataset.py                  # Feature engineering and PyTorch Dataset class
│   ├── main.py                     # Main script for training the model
│   ├── model.py                    # LSTM model definition
│   ├── eval.py                     # Script for evaluating the trained model
│   └── preprocess.py               # Script for initial data cleaning and splitting
├── README.md                       # This file
└── requirements.txt                # Project dependencies
```

## Setup

1.  **Clone the repository** (if you haven't already).

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Workflow

The pipeline is a two-step process:

1.  **Preprocess the Data**:
    This script cleans the raw data, downsamples it to an hourly frequency, and splits it into `train.csv`, `val.csv`, and `test.csv`. This only needs to be run once.
    ```bash
    python -m src.preprocess
    ```

2.  **Train the Model**:
    This script handles feature engineering, data scaling, and the main training loop. It saves the best model and the data scaler to the `problem2/results/` directory.
    ```bash
    python -m src.main
    ```

3.  **Evaluate the Model**:
    This script loads the test set, the saved model, and the scaler to report the final performance metrics (MAE, RMSE, MAPE) on unseen data.
    ```bash
    python -m src.eval
    ```