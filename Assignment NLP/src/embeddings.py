import numpy as np
import joblib

from gensim.models import Word2Vec

from sentence_transformers import SentenceTransformer


class EmbeddingExtractor:

    """
    Handles:
    1. Word2Vec
    2. BERT Sentence Embeddings
    """

    def __init__(

        self,

        vector_size=100,

        window=5,

        min_count=1,

        workers=4,

        epochs=20,

        bert_model="all-MiniLM-L6-v2"

    ):

        # ----------------------------
        # Word2Vec Parameters
        # ----------------------------

        self.vector_size = vector_size

        self.window = window

        self.min_count = min_count

        self.workers = workers

        self.epochs = epochs

        self.model = None

        # ----------------------------
        # BERT Model
        # ----------------------------

        print()

        print("=" * 60)
        print("Loading BERT Model...")
        print("=" * 60)

        self.bert = SentenceTransformer(bert_model)

    # =====================================================
    # WORD2VEC
    # =====================================================

    def train_word2vec(self, sentences):

        tokenized = [

            sentence.split()

            for sentence in sentences

        ]

        self.model = Word2Vec(

            sentences=tokenized,

            vector_size=self.vector_size,

            window=self.window,

            min_count=self.min_count,

            workers=self.workers,

            epochs=self.epochs

        )

        print()

        print("=" * 60)
        print("WORD2VEC MODEL TRAINED")
        print("=" * 60)

        print("Vocabulary Size :", len(self.model.wv))

        return self.model

    # =====================================================
    # Sentence Vector
    # =====================================================

    def sentence_vector(self, sentence):

        vectors = []

        for word in sentence.split():

            if word in self.model.wv:

                vectors.append(self.model.wv[word])

        if len(vectors) == 0:

            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    # =====================================================
    # Generate Word2Vec Embeddings
    # =====================================================

    def generate_word2vec_embeddings(self, sentences):

        return np.array(

            [

                self.sentence_vector(sentence)

                for sentence in sentences

            ]

        )

    # =====================================================
    # Generate BERT Embeddings
    # =====================================================

    def generate_bert_embeddings(

        self,

        sentences,

        batch_size=32

    ):

        embeddings = self.bert.encode(

            list(sentences),

            batch_size=batch_size,

            show_progress_bar=True,

            convert_to_numpy=True

        )

        return embeddings

    # =====================================================
    # Save Word2Vec Model
    # =====================================================

    def save_model(self, path):

        self.model.save(str(path))

    # =====================================================
    # Save NumPy
    # =====================================================

    def save_embeddings(

        self,

        embeddings,

        path

    ):

        np.save(str(path), embeddings)

    # =====================================================
    # Save Joblib
    # =====================================================

    def save_with_joblib(

        self,

        data,

        path

    ):

        joblib.dump(data, str(path))