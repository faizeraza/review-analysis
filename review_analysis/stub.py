import cleantext
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle
def predictsentiment(text):
    cleaned = cleantext.clean(text, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True)
    countv = CountVectorizer()
    df = pd.DataFrame({"target":[cleaned]})
    print(df)
    reviews_countv = countv.fit_transform(df["target"])
    df_alexa2 = pd.DataFrame(reviews_countv.toarray())
    print(type(df_alexa2))
    print(df_alexa2)
    ls = os.listdir(r"review_analysis/models")
    for model in ls[1:]:
        filename = f"review_analysis/models/{model}"
        loaded_model = pickle.load(open(filename, 'rb'))
        prediction = loaded_model.predict(df_alexa2)
        print(prediction)
        break

predictsentiment("As the output shows, after tokenization, all the words in the response are separated into smaller units (i.e., tokens). In the following analysis, each word in the corpus will be handled separately in the Word2Vec process. Now, let’s repeat the same procedure for the entire corpus.")