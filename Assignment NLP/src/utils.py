import random
import numpy as np


def print_heading(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_before_after(original_df, processed_df, rows=5):

    comparison = original_df[["text"]].copy()

    comparison["clean_text"] = processed_df["clean_text"]

    print(comparison.head(rows))


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)