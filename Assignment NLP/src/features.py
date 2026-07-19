from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

import joblib


class FeatureExtractor:

    """
    Traditional NLP Feature Extraction
    """

    def __init__(self):

        # One Hot Encoder
        self.onehot_vectorizer = CountVectorizer(
            binary=True
        )

        # Bag of Words
        self.bow_vectorizer = CountVectorizer()

        # Bigram
        self.bigram_vectorizer = CountVectorizer(
            ngram_range=(1, 2)
        )

        # Trigram
        self.trigram_vectorizer = CountVectorizer(
            ngram_range=(1, 3)
        )

        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer()

    # ---------------------------------------------------------

    def one_hot(self, train, validation, test):

        X_train = self.onehot_vectorizer.fit_transform(train)

        X_validation = self.onehot_vectorizer.transform(validation)

        X_test = self.onehot_vectorizer.transform(test)

        return X_train, X_validation, X_test

    # ---------------------------------------------------------

    def bag_of_words(self, train, validation, test):

        X_train = self.bow_vectorizer.fit_transform(train)

        X_validation = self.bow_vectorizer.transform(validation)

        X_test = self.bow_vectorizer.transform(test)

        return X_train, X_validation, X_test

    # ---------------------------------------------------------

    def bigram(self, train, validation, test):

        X_train = self.bigram_vectorizer.fit_transform(train)

        X_validation = self.bigram_vectorizer.transform(validation)

        X_test = self.bigram_vectorizer.transform(test)

        return X_train, X_validation, X_test

    # ---------------------------------------------------------

    def trigram(self, train, validation, test):

        X_train = self.trigram_vectorizer.fit_transform(train)

        X_validation = self.trigram_vectorizer.transform(validation)

        X_test = self.trigram_vectorizer.transform(test)

        return X_train, X_validation, X_test

    # ---------------------------------------------------------

    def tfidf(self, train, validation, test):

        X_train = self.tfidf_vectorizer.fit_transform(train)

        X_validation = self.tfidf_vectorizer.transform(validation)

        X_test = self.tfidf_vectorizer.transform(test)

        return X_train, X_validation, X_test

    # ---------------------------------------------------------

    def save_vectorizer(self, vectorizer, file_path):

        joblib.dump(
            vectorizer,
            file_path
        )

    # ---------------------------------------------------------

    def print_feature_info(self, name, X, vectorizer):

        print()

        print("=" * 60)
        print(name)
        print("=" * 60)

        print("Shape:", X.shape)

        print("Total Features:", len(vectorizer.get_feature_names_out()))

        print()

        print("First 20 Features:")

        print(vectorizer.get_feature_names_out()[:20])