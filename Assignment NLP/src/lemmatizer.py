import spacy


class LemmatizerProcessor:

    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")

    def lemmatize(self, text):

        doc = self.nlp(text)

        return " ".join(
            token.lemma_
            for token in doc
        )

    def apply(self, df):

        df["lemma"] = df["clean_text"].apply(
            self.lemmatize
        )

        return df