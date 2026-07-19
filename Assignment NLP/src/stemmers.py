from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    SnowballStemmer,
    RegexpStemmer
)


class StemmerProcessor:

    def __init__(self):

        self.porter = PorterStemmer()

        self.lancaster = LancasterStemmer()

        self.snowball = SnowballStemmer("english")

        # Remove common suffixes
        self.regexp = RegexpStemmer(
            r"(ing|ed|ly|es|s)$"
        )

    def porter_stem(self, text):

        words = text.split()

        return " ".join(
            self.porter.stem(word)
            for word in words
        )

    def lancaster_stem(self, text):

        words = text.split()

        return " ".join(
            self.lancaster.stem(word)
            for word in words
        )

    def snowball_stem(self, text):

        words = text.split()

        return " ".join(
            self.snowball.stem(word)
            for word in words
        )

    def regexp_stem(self, text):

        words = text.split()

        return " ".join(
            self.regexp.stem(word)
            for word in words
        )

    def apply_all(self, df):

        df["stem_porter"] = df["clean_text"].apply(
            self.porter_stem
        )

        df["stem_lancaster"] = df["clean_text"].apply(
            self.lancaster_stem
        )

        df["stem_snowball"] = df["clean_text"].apply(
            self.snowball_stem
        )

        df["stem_regexp"] = df["clean_text"].apply(
            self.regexp_stem
        )

        return df