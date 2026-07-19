import pandas as pd

from src.config import TRAIN_FILE, VALIDATION_FILE, TEST_FILE


class DatasetLoader:

    def __init__(self):
        self.train = None
        self.validation = None
        self.test = None
        self.full_dataset = None

    def load_data(self):
        """Load all datasets."""

        self.train = pd.read_csv(TRAIN_FILE)
        self.validation = pd.read_csv(VALIDATION_FILE)
        self.test = pd.read_csv(TEST_FILE)

        print("Datasets Loaded Successfully")

        return self.train, self.validation, self.test

    def combine_datasets(self):
        """Combine all datasets."""

        self.full_dataset = pd.concat(
            [self.train, self.validation, self.test],
            ignore_index=True
        )

        return self.full_dataset

    def dataset_info(self, df, name):

        print("=" * 60)
        print(name)
        print("=" * 60)

        print(df.info())

        print("\nShape:")
        print(df.shape)

        print("\nColumns:")
        print(df.columns.tolist())

    def missing_values(self, df):

        return df.isnull().sum()

    def duplicate_rows(self, df):

        return df.duplicated().sum()

    def class_distribution(self, df):

        return df["label"].value_counts().sort_index()

    def descriptive_statistics(self, df):

        return df.describe(include="all")
    
    def preprocess_dataframe(self, df, preprocessor):

        processed = df.copy()

        processed["clean_text"] = processed["text"].apply(
            preprocessor.preprocess
        )

        return processed


    def save_processed_dataset(self, df, file_path):

        df.to_csv(file_path, index=False)

        print(f"Saved: {file_path}")

    def preview_columns(self, df, rows=5):

        columns = [
            "text",
            "clean_text",
            "stem_porter",
            "stem_lancaster",
            "stem_snowball",
            "stem_regexp",
            "lemma"
        ]

        print(df[columns].head(rows))