from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_CSV_PATH = RAW_DATA_DIR / "Original training DB i1 cluster.csv"
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "training_db.csv"
VERIFY_CSV_PATH = PROCESSED_DATA_DIR / "verification_db.csv"

RANDOM_STATE = 42
LABEL_COL = "Label"   # change to "label" if needed after checking the CSV
POS_LABEL = 1
NEG_LABEL = 0
