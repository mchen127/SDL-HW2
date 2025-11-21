import torch
import os

# -- File Paths --
# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data path
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "jena_climate_2009_2016.csv")

# Processed data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_CSV_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")

# -- Data Configuration --
TARGET_FEATURE = "T (degC)"

# -- Preprocessing & Splitting Configuration --
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
# TEST_SPLIT is implicitly 1 - TRAIN_SPLIT - VAL_SPLIT

# -- Dataloader Configuration --
INPUT_WIDTH = 24  # Input sequence length (24 hours of history)
LABEL_WIDTH = 1   # Output sequence length (predict 1 hour ahead)
SHIFT = 1         # Time offset between input and output

# -- Model Hyperparameters --
INPUT_SIZE = 19   # Number of features after preprocessing
HIDDEN_SIZE = 32  # Number of features in the hidden state of the LSTM
NUM_LAYERS = 1    # Number of recurrent layers
OUTPUT_SIZE = 1   # We are predicting a single value

# -- Training Configuration --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 256
PATIENCE = 5 # For early stopping

# -- Artifacts Path --
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _artifact_suffix(input_width: int, label_width: int, shift: int) -> str:
    return f"iw{input_width}_lw{label_width}_sh{shift}"


def model_path(input_width: int = INPUT_WIDTH, label_width: int = LABEL_WIDTH, shift: int = SHIFT) -> str:
    """Return the path for the trained model corresponding to the supplied window config."""
    suffix = _artifact_suffix(input_width, label_width, shift)
    return os.path.join(RESULTS_DIR, f"best_model_{suffix}.pth")


def scaler_path(input_width: int = INPUT_WIDTH, label_width: int = LABEL_WIDTH, shift: int = SHIFT) -> str:
    suffix = _artifact_suffix(input_width, label_width, shift)
    return os.path.join(RESULTS_DIR, f"scaler_{suffix}.pkl")


def plot_path(input_width: int = INPUT_WIDTH, label_width: int = LABEL_WIDTH, shift: int = SHIFT) -> str:
    suffix = _artifact_suffix(input_width, label_width, shift)
    return os.path.join(RESULTS_DIR, f"test_predictions_{suffix}.png")


def results_csv_path(input_width: int = INPUT_WIDTH, label_width: int = LABEL_WIDTH, shift: int = SHIFT) -> str:
    suffix = _artifact_suffix(input_width, label_width, shift)
    return os.path.join(RESULTS_DIR, f"test_results_{suffix}.csv")
