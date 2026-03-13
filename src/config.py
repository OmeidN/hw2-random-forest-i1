"""
Project configuration for HW2 Random Forest i1 cluster pipeline.
Uses pathlib for cross-platform paths (Windows/Mac/Linux).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and directories
# ---------------------------------------------------------------------------
# Resolve to absolute path so it works from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "outputs"

# Reports and figures
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# CSV file paths
# ---------------------------------------------------------------------------
RAW_CSV_PATH = RAW_DATA_DIR / "Original training DB i1 cluster.csv"
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "training_db.csv"
VERIFY_CSV_PATH = PROCESSED_DATA_DIR / "verification_db.csv"

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Label column and class values
# The code detects "label" or "Label" via data_utils.detect_label_column()
# ---------------------------------------------------------------------------
LABEL_COL = "label"   # Expected name; use detect_label_column(df) to get actual
POS_LABEL = 1          # i1 cluster
NEG_LABEL = 0          # non-i1

# ---------------------------------------------------------------------------
# Optional: if your CSV uses "Label" (capital L), set this after loading:
# label_col = detect_label_column(df)  # returns "label" or "Label"
# ---------------------------------------------------------------------------
