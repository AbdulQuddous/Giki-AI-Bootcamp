# from src.dataset import DatasetLoader
# from src.preprocess import TextPreprocessor
# from src.utils import print_heading, show_before_after
# from src.config import (
#     PROCESSED_DATA_DIR,
# )


# class Pipeline:

#     def __init__(self):

#         self.loader = DatasetLoader()

#         self.preprocessor = TextPreprocessor()

#     def run(self):

#         train, validation, test = self.loader.load_data()

#         print_heading("PREPROCESSING TRAIN")

#         train_processed = self.loader.preprocess_dataframe(
#             train,
#             self.preprocessor,
#         )

#         validation_processed = self.loader.preprocess_dataframe(
#             validation,
#             self.preprocessor,
#         )

#         test_processed = self.loader.preprocess_dataframe(
#             test,
#             self.preprocessor,
#         )

#         self.loader.save_processed_dataset(
#             train_processed,
#             PROCESSED_DATA_DIR /
#             "emotion_train_processed.csv"
#         )

#         self.loader.save_processed_dataset(
#             validation_processed,
#             PROCESSED_DATA_DIR /
#             "emotion_validation_processed.csv"
#         )

#         self.loader.save_processed_dataset(
#             test_processed,
#             PROCESSED_DATA_DIR /
#             "emotion_test_processed.csv"
#         )

#         print_heading("BEFORE vs AFTER")

#         show_before_after(
#             train,
#             train_processed
#         )


import joblib

from src.dataset import DatasetLoader
from src.preprocess import TextPreprocessor
from src.stemmers import StemmerProcessor
from src.lemmatizer import LemmatizerProcessor
from src.features import FeatureExtractor
from src.embeddings import EmbeddingExtractor

from src.utils import print_heading

from src.config import (
    PROCESSED_DATA_DIR,

    ONEHOT_DIR,
    BOW_DIR,
    NGRAM_DIR,
    TFIDF_DIR,

    WORD2VEC_MODEL,
    WORD2VEC_TRAIN,
    WORD2VEC_VALIDATION,
    WORD2VEC_TEST,

    BERT_TRAIN,
    BERT_VALIDATION,
    BERT_TEST
)


class Pipeline:

    def __init__(self):

        # Dataset Loader
        self.loader = DatasetLoader()

        # Text Processing
        self.preprocessor = TextPreprocessor()

        self.stemmer = StemmerProcessor()

        self.lemmatizer = LemmatizerProcessor()

        # Feature Engineering
        self.features = FeatureExtractor()

        # Embeddings
        self.embedding = EmbeddingExtractor()

    # =====================================================
    # Process Dataset
    # =====================================================

    def process_dataset(self, df):

        df = self.loader.preprocess_dataframe(
            df,
            self.preprocessor
        )

        df = self.stemmer.apply_all(df)

        df = self.lemmatizer.apply(df)

        return df

    # =====================================================
    # Main Pipeline
    # =====================================================

    def run(self):

        # =====================================================
        # Load Dataset
        # =====================================================

        train, validation, test = self.loader.load_data()

        # =====================================================
        # Preprocessing
        # =====================================================

        print_heading("PROCESSING DATASETS")

        train = self.process_dataset(train)

        validation = self.process_dataset(validation)

        test = self.process_dataset(test)

        # =====================================================
        # Save Processed CSV Files
        # =====================================================

        self.loader.save_processed_dataset(
            train,
            PROCESSED_DATA_DIR / "emotion_train_processed.csv"
        )

        self.loader.save_processed_dataset(
            validation,
            PROCESSED_DATA_DIR / "emotion_validation_processed.csv"
        )

        self.loader.save_processed_dataset(
            test,
            PROCESSED_DATA_DIR / "emotion_test_processed.csv"
        )

        # =====================================================
        # Preview
        # =====================================================

        print_heading("SAMPLE OUTPUT")

        self.loader.preview_columns(train)

        # =====================================================
        # Prepare Text
        # =====================================================

        train_text = train["clean_text"]

        validation_text = validation["clean_text"]

        test_text = test["clean_text"]

            # =====================================================
        # ONE HOT ENCODING
        # =====================================================

        print_heading("ONE HOT ENCODING")

        onehot_train, onehot_validation, onehot_test = self.features.one_hot(
            train_text,
            validation_text,
            test_text
        )

        joblib.dump(
            (
                onehot_train,
                onehot_validation,
                onehot_test
            ),
            ONEHOT_DIR / "onehot.pkl"
        )

        self.features.save_vectorizer(
            self.features.onehot_vectorizer,
            ONEHOT_DIR / "vectorizer.pkl"
        )

        self.features.print_feature_info(
            "One Hot Encoding",
            onehot_train,
            self.features.onehot_vectorizer
        )

        # =====================================================
        # BAG OF WORDS
        # =====================================================

        print_heading("BAG OF WORDS")

        bow_train, bow_validation, bow_test = self.features.bag_of_words(
            train_text,
            validation_text,
            test_text
        )

        joblib.dump(
            (
                bow_train,
                bow_validation,
                bow_test
            ),
            BOW_DIR / "bow.pkl"
        )

        self.features.save_vectorizer(
            self.features.bow_vectorizer,
            BOW_DIR / "vectorizer.pkl"
        )

        self.features.print_feature_info(
            "Bag of Words",
            bow_train,
            self.features.bow_vectorizer
        )

        # =====================================================
        # BIGRAM
        # =====================================================

        print_heading("BIGRAM")

        bigram_train, bigram_validation, bigram_test = self.features.bigram(
            train_text,
            validation_text,
            test_text
        )

        joblib.dump(
            (
                bigram_train,
                bigram_validation,
                bigram_test
            ),
            NGRAM_DIR / "bigram.pkl"
        )

        self.features.save_vectorizer(
            self.features.bigram_vectorizer,
            NGRAM_DIR / "bigram_vectorizer.pkl"
        )

        self.features.print_feature_info(
            "Bigram",
            bigram_train,
            self.features.bigram_vectorizer
        )

        # =====================================================
        # TRIGRAM
        # =====================================================

        print_heading("TRIGRAM")

        trigram_train, trigram_validation, trigram_test = self.features.trigram(
            train_text,
            validation_text,
            test_text
        )

        joblib.dump(
            (
                trigram_train,
                trigram_validation,
                trigram_test
            ),
            NGRAM_DIR / "trigram.pkl"
        )

        self.features.save_vectorizer(
            self.features.trigram_vectorizer,
            NGRAM_DIR / "trigram_vectorizer.pkl"
        )

        self.features.print_feature_info(
            "Trigram",
            trigram_train,
            self.features.trigram_vectorizer
        )

        # =====================================================
        # TF-IDF
        # =====================================================

        print_heading("TF-IDF")

        tfidf_train, tfidf_validation, tfidf_test = self.features.tfidf(
            train_text,
            validation_text,
            test_text
        )

        joblib.dump(
            (
                tfidf_train,
                tfidf_validation,
                tfidf_test
            ),
            TFIDF_DIR / "tfidf.pkl"
        )

        self.features.save_vectorizer(
            self.features.tfidf_vectorizer,
            TFIDF_DIR / "vectorizer.pkl"
        )

        self.features.print_feature_info(
            "TF-IDF",
            tfidf_train,
            self.features.tfidf_vectorizer
        )

        print_heading("FEATURE ENGINEERING COMPLETED")
    
            # =====================================================
        # WORD2VEC EMBEDDINGS
        # =====================================================

        print_heading("WORD2VEC EMBEDDINGS")

        # Train Word2Vec Model
        self.embedding.train_word2vec(train_text)

        # Generate Embeddings
        word2vec_train = self.embedding.generate_word2vec_embeddings(
            train_text
        )

        word2vec_validation = self.embedding.generate_word2vec_embeddings(
            validation_text
        )

        word2vec_test = self.embedding.generate_word2vec_embeddings(
            test_text
        )

        # Save Model
        self.embedding.save_model(
            WORD2VEC_MODEL
        )

        # Save Embeddings
        self.embedding.save_embeddings(
            word2vec_train,
            WORD2VEC_TRAIN
        )

        self.embedding.save_embeddings(
            word2vec_validation,
            WORD2VEC_VALIDATION
        )

        self.embedding.save_embeddings(
            word2vec_test,
            WORD2VEC_TEST
        )

        print()

        print("=" * 60)
        print("WORD2VEC RESULTS")
        print("=" * 60)

        print("Train Shape      :", word2vec_train.shape)
        print("Validation Shape :", word2vec_validation.shape)
        print("Test Shape       :", word2vec_test.shape)

        print()

        print("Sample Vector (First 10 Values):")

        print(word2vec_train[0][:10])

        # =====================================================
        # BERT EMBEDDINGS
        # =====================================================

        print_heading("BERT EMBEDDINGS")

        bert_train = self.embedding.generate_bert_embeddings(
            train_text
        )

        bert_validation = self.embedding.generate_bert_embeddings(
            validation_text
        )

        bert_test = self.embedding.generate_bert_embeddings(
            test_text
        )

        # Save Embeddings

        self.embedding.save_embeddings(
            bert_train,
            BERT_TRAIN
        )

        self.embedding.save_embeddings(
            bert_validation,
            BERT_VALIDATION
        )

        self.embedding.save_embeddings(
            bert_test,
            BERT_TEST
        )

        print()

        print("=" * 60)
        print("BERT RESULTS")
        print("=" * 60)

        print("Train Shape      :", bert_train.shape)
        print("Validation Shape :", bert_validation.shape)
        print("Test Shape       :", bert_test.shape)

        print()

        print("Sample Vector (First 10 Values):")

        print(bert_train[0][:10])

        # =====================================================
        # PHASE COMPLETED
        # =====================================================

        print_heading("PHASE 5 COMPLETED")