
# document = "I love Natural Language Processing."

# print(document)

# corpus = [
#     "I love NLP.",
#     "Python is amazing.",
#     "NLP makes computers understand language."
# ]

# print("Corpus Size:", len(corpus))

# print("\nDocuments:\n")

# for i, doc in enumerate(corpus, start=1):
#     print(f"Document {i}: {doc}")

# vocabulary = set()
# for document in corpus:
#     for word in document.split():
#         vocabulary.add(word)

# print("Vocabulary:\n")
# print(vocabulary)

# print("\nVocabulary Size:", len(vocabulary))

# import pandas as pd

# df = pd.read_csv('F:/Giki Ai bootcamp/Nlp/reviews.csv')

# # print(df.head())
# # print(df.shape)
# # print(df.columns)

# list_of_column_names = list(df.columns)
# print('List of column names : ', list_of_column_names)
# df = df.rename(columns={df.columns[2]: 'Text'})
# df.columns
# print(df['Text'][:5])

# print(df['Text'].str.lower())

import re

txt = "The rain in GIKI"
x = re.search("^The.*GIKI$", txt)
print(x)


text = "apple mango zebra"

result = re.findall("[a-m]", text)

print(result)
