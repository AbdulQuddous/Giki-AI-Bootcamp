
document = "I love Natural Language Processing."

print(document)

corpus = [
    "I love NLP.",
    "Python is amazing.",
    "NLP makes computers understand language."
]

print("Corpus Size:", len(corpus))

print("\nDocuments:\n")

for i, doc in enumerate(corpus, start=1):
    print(f"Document {i}: {doc}")

vocabulary = set()
for document in corpus:
    for word in document.split():
        vocabulary.add(word)

print("Vocabulary:\n")
print(vocabulary)

print("\nVocabulary Size:", len(vocabulary))
