from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUT_DIR / "figures"

CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrix"

COMPARISON_TABLE_DIR = OUTPUT_DIR / "comparison_tables"

REPORT_DIR = OUTPUT_DIR / "reports"

MODEL_DIR = OUTPUT_DIR / "trained_models"

# ==========================================================
# Dataset Files
# ==========================================================

TRAIN_FILE = RAW_DATA_DIR / "emotion_train.csv"

VALIDATION_FILE = RAW_DATA_DIR / "emotion_validation.csv"

TEST_FILE = RAW_DATA_DIR / "emotion_test.csv"

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# Feature Engineering Directories
# ==========================================================

FEATURE_DIR = OUTPUT_DIR / "features"

ONEHOT_DIR = FEATURE_DIR / "onehot"

BOW_DIR = FEATURE_DIR / "bow"

NGRAM_DIR = FEATURE_DIR / "ngram"

TFIDF_DIR = FEATURE_DIR / "tfidf"

# ==========================================================
# Embedding Directories
# ==========================================================

EMBEDDING_DIR = OUTPUT_DIR / "embeddings"

WORD2VEC_DIR = EMBEDDING_DIR / "word2vec"

BERT_DIR = EMBEDDING_DIR / "bert"

# ==========================================================
# Word2Vec Files
# ==========================================================

WORD2VEC_MODEL = WORD2VEC_DIR / "word2vec.model"

WORD2VEC_TRAIN = WORD2VEC_DIR / "train.npy"

WORD2VEC_VALIDATION = WORD2VEC_DIR / "validation.npy"

WORD2VEC_TEST = WORD2VEC_DIR / "test.npy"

# ==========================================================
# BERT Files
# ==========================================================

BERT_TRAIN = BERT_DIR / "train.npy"

BERT_VALIDATION = BERT_DIR / "validation.npy"

BERT_TEST = BERT_DIR / "test.npy"

# ==========================================================
# Create Directories Automatically
# ==========================================================

DIRECTORIES = [

    FIGURES_DIR,

    CONFUSION_MATRIX_DIR,

    COMPARISON_TABLE_DIR,

    REPORT_DIR,

    MODEL_DIR,

    FEATURE_DIR,

    ONEHOT_DIR,

    BOW_DIR,

    NGRAM_DIR,

    TFIDF_DIR,

    EMBEDDING_DIR,

    WORD2VEC_DIR,

    BERT_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)