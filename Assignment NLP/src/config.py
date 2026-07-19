from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output Directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUT_DIR / "figures"
CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrix"
COMPARISON_TABLE_DIR = OUTPUT_DIR / "comparison_tables"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "trained_models"

# Dataset Files
TRAIN_FILE = RAW_DATA_DIR / "emotion_train.csv"
VALIDATION_FILE = RAW_DATA_DIR / "emotion_validation.csv"
TEST_FILE = RAW_DATA_DIR / "emotion_test.csv"

# Random Seed
RANDOM_STATE = 42