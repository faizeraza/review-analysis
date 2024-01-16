import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import confusion_matrix as cm
import pickle
import cleantext
from nltk import word_tokenize

st.title('Sentiment Analyser App')
st.write('Welcome to Our sentiment analysis app! \t bellow are the accuracy of different model for our trained dataset.')

def plot_acaAndcm(model_ac,model_cm,df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(model_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.write(model,model_ac)
    st.write(fig,df)

def predictsentiment(text):
    cleaned = cleantext.clean(text, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True)
    tokenaized = word_tokenize(cleaned)
    print(tokenaized)
# Bellow code is for test case accuracy_score check
X_test = pd.read_csv("review_analysis/data/processed/X_test.csv")
y_test = pd.read_csv("review_analysis/data/processed/y_test.csv")
ls = os.listdir(r"review_analysis/models")
for model in ls:
    filename = f"review_analysis/models/{model}"
    loaded_model = pickle.load(open(filename, 'rb'))
    prediction = pd.DataFrame({"prediction":loaded_model.predict(X_test)})
    # print(prediction)
    df = pd.concat([prediction["prediction"],y_test['feedback']],axis=1)
    # st.write(df)
    model_ac = ac(y_test['feedback'], prediction)
    model_cm = cm(y_test['feedback'],prediction)
    plot_acaAndcm(model_ac,model_cm,df)
    
