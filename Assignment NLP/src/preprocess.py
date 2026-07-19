import re
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))

    def to_lowercase(self, text):

        return text.lower()

    def remove_html(self, text):

        return re.sub(r"<.*?>", "", text)

    def remove_urls(self, text):

        return re.sub(r"http\S+|www\S+", "", text)

    def remove_mentions(self, text):

        return re.sub(r"@\w+", "", text)

    def remove_hashtags(self, text):

        return re.sub(r"#", "", text)

    def remove_numbers(self, text):

        return re.sub(r"\d+", "", text)

    def remove_punctuation(self, text):

        return text.translate(
            str.maketrans("", "", string.punctuation)
        )

    def remove_extra_spaces(self, text):

        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text):

        return word_tokenize(text)

    def remove_short_words(self, tokens):

        filtered = []

        for word in tokens:

            if len(word) > 2 or word in ["i", "a"]:
                filtered.append(word)

        return filtered

    def spelling_correction(self, tokens):

        corrected = []

        for word in tokens:

            corrected.append(str(TextBlob(word).correct()))

        return corrected

    def remove_stopwords(self, tokens):

        return [
            word
            for word in tokens
            if word not in self.stop_words
        ]

    def preprocess(self, text):

        text = self.to_lowercase(text)

        text = self.remove_html(text)

        text = self.remove_urls(text)

        text = self.remove_mentions(text)

        text = self.remove_hashtags(text)

        text = self.remove_numbers(text)

        text = self.remove_punctuation(text)

        text = self.remove_extra_spaces(text)

        tokens = self.tokenize(text)

        tokens = self.remove_short_words(tokens)

        tokens = self.spelling_correction(tokens)

        tokens = self.remove_stopwords(tokens)

        clean_text = " ".join(tokens)

        return clean_text