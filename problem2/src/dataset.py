import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import src.config as config


class JenaClimateDataset(Dataset):
    """PyTorch Dataset for the Jena Climate data."""

    def __init__(self, data, input_width, label_width, shift):
        """
        Args:
            data (pd.DataFrame): DataFrame with scaled and feature-engineered data.
            input_width (int): Number of timesteps in the input sequence.
            label_width (int): Number of timesteps in the label sequence.
            shift (int): The offset between the end of an input window and
                         the start of the label window.
        """
        self.data = data.to_numpy(dtype=np.float32)
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift

        # Find the index of the target column
        self.target_col_index = data.columns.get_loc(config.TARGET_FEATURE)

    def __len__(self):
        return len(self.data) - self.total_window_size + 1

    def __getitem__(self, idx):
        """
        Returns a single window of data (input features and corresponding label).
        """
        window = self.data[idx : idx + self.total_window_size]

        # Input features are the first `input_width` timesteps
        inputs = window[: self.input_width, :]

        # Label is the target feature at the end of the window
        label_start_idx = self.total_window_size - self.label_width
        labels = window[label_start_idx:, self.target_col_index]

        return torch.from_numpy(inputs), torch.from_numpy(labels)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers cyclical features for the Jena Climate dataset.
    """
    # Ensure the index is a DatetimeIndex for time-based feature engineering
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date Time'] = pd.to_datetime(df['Date Time'])
        df.set_index('Date Time', inplace=True)

    # Decompose wind direction into cyclical components
    wd_rad = df.pop('wd (deg)') * np.pi / 180
    df["Wx"] = np.cos(wd_rad)
    df["Wy"] = np.sin(wd_rad)

    # Decompose time into cyclical components
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))
    
    return df