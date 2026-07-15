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