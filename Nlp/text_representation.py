import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'text': ['he cooks food','food cooks good', \
                            'food smells good', 'food is food'], \
                   'output':[0,1,1,0]})
print(df)


encoder = OneHotEncoder(sparse_output=False)

one_hot = encoder.fit_transform(df[['text']])

print(one_hot)

print(encoder.get_feature_names_out())


words_array = np.array(['I', "Love", "NLP", "is", "Fun", "NLP"]).reshape(-1, 1)

encoder = OneHotEncoder()

one_hot = encoder.fit_transform(words_array)

print(one_hot)



all_words = []

for sentence in df['text']:
    words = sentence.split()
    all_words.extend(words)


# 2. Get unique vocabulary
vocabulary = sorted(set(all_words))

print("\nVocabulary:")
print(vocabulary)

word_dataframe = pd.DataFrame(vocabulary, columns=['word'])

print("\nWord DataFrame:")
print(word_dataframe)


encoder = OneHotEncoder(sparse_output=False)

word_one_hot = encoder.fit_transform(word_dataframe[['word']])

print("\nOne Hot Encoding:")
print(word_one_hot)

print("\nFeature Names:")
print(encoder.get_feature_names_out())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr

sns.set_style("whitegrid")

def one_hot_visualizer(text):

    # Input
    categories = [x.strip() for x in text.split(",") if x.strip()]

    if len(categories) == 0:
        return None, None

    # Original DataFrame
    df = pd.DataFrame({"Category": categories})

    # One Hot Encoding
    encoded = pd.get_dummies(df, columns=["Category"], dtype=int)

    # Heatmap
    cols = [c for c in encoded.columns if c.startswith("Category_")]

    plt.figure(figsize=(max(6,len(cols)), max(3,len(df)/2)))

    sns.heatmap(
        encoded[cols],
        annot=True,
        cmap="YlGnBu",
        linewidths=1,
        cbar=False,
        fmt="d"
    )

    plt.title("One-Hot Encoding Matrix")
    plt.xlabel("Encoded Features")
    plt.ylabel("Samples")
    plt.tight_layout()

    return encoded, plt.gcf()


demo = gr.Interface(
    fn=one_hot_visualizer,
    inputs=gr.Textbox(
        lines=2,
        label="Enter Categories (comma separated)",
        placeholder="Apple, Banana, Apple, Mango, Banana, Orange"
    ),
    outputs=[
        gr.Dataframe(label="One-Hot Encoded Data"),
        gr.Plot(label="Visualization")
    ],
    title="🎓 One-Hot Encoding (OHE) Visualizer",
    description="""
Type any categorical values separated by commas.

Examples:

Apple, Banana, Apple, Mango

Cat, Dog, Bird, Dog, Cat

Red, Blue, Green, Blue

AI, Web, AI, Cyber, Data Science
"""
)

demo.launch()