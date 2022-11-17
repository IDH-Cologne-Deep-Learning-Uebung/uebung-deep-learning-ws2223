import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Test NumPy
print(np.zeros((2, 3)))

# Test Pandas
titanic = pd.read_csv("titanic.csv")
print(titanic.dtypes[1:2])

# Test scikit-learn
fo = open("wiki.txt", encoding="UTF-8")
lines = [line for line in fo.readlines()]
fo.close()

vectorizer = CountVectorizer(max_features=5000)
vectorizer.fit(lines)
vectors = vectorizer.transform(lines)
print(vectors[1])