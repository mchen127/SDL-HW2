import pandas as pd
import src.config as config
import os

def create_processed_datasets():
    """
    Loads the raw Jena Climate dataset, performs initial cleaning and
    downsampling, splits it into train, validation, and test sets,
    and saves them to the data directory.
    """
    print("Loading raw data...")
    df = pd.read_csv(config.CSV_PATH)
    print(f"NaNs after loading: {df.isnull().sum().sum()}")

    # --- Data Cleaning and Resampling ---
    print("Cleaning and downsampling data...")
    # Parse dates and set as index
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df.set_index("Date Time", inplace=True)

    # Replace erroneous wind velocity values BEFORE resampling
    df["wv (m/s)"] = df["wv (m/s)"].replace(-9999.0, 0.0)
    df["max. wv (m/s)"] = df["max. wv (m/s)"].replace(-9999.0, 0.0)
    print(f"NaNs after cleaning -9999s: {df.isnull().sum().sum()}")

    # Resample to hourly frequency, taking the mean
    df = df.resample("h").mean()
    print(f"NaNs after resampling: {df.isnull().sum().sum()}")

    # Forward-fill any missing values that resulted from resampling
    df.ffill(inplace=True)
    print(f"NaNs after ffill: {df.isnull().sum().sum()}")
    # --- Data Splitting ---
    print("Splitting data into train, validation, and test sets...")
    n = len(df)
    train_df = df[0 : int(n * config.TRAIN_SPLIT)]
    val_df = df[int(n * config.TRAIN_SPLIT) : int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))]
    test_df = df[int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT)) :]

    # --- Save Datasets ---
    print(f"Saving training set to {config.TRAIN_CSV_PATH}")
    train_df.to_csv(config.TRAIN_CSV_PATH)
    print(f"Saving validation set to {config.VAL_CSV_PATH}")
    val_df.to_csv(config.VAL_CSV_PATH)
    print(f"Saving test set to {config.TEST_CSV_PATH}")
    test_df.to_csv(config.TEST_CSV_PATH)
    print("Preprocessing complete.")

if __name__ == "__main__":
    create_processed_datasets()