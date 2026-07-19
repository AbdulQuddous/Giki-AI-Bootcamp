import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR


class DatasetVisualizer:

    def plot_class_distribution(self, df):

        plt.figure(figsize=(12,6))

        sns.countplot(
            x="label",
            data=df,
            order=sorted(df["label"].unique())
        )

        plt.title("Class Distribution")

        plt.xlabel("Emotion Label")

        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(FIGURES_DIR / "class_distribution.png")

        plt.show()

    def plot_text_length(self, df):

        lengths = df["text"].str.split().apply(len)

        plt.figure(figsize=(10,6))

        plt.hist(lengths, bins=40)

        plt.title("Tweet Length Distribution")

        plt.xlabel("Words")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(FIGURES_DIR / "tweet_length_distribution.png")

        plt.show()