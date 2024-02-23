import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import confusion_matrix as cm
import pickle
from nltk import word_tokenize

# theme_options = ["Light", "Dark"]
# selected_theme = st.selectbox("Select Theme", theme_options)

# # Set the theme based on user selection
# if selected_theme == "Light":
#     st.set_page_config(theme="light")
# elif selected_theme == "Dark":
#     st.set_page_config(theme="dark")


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

# Bellow code is for test case accuracy_score check
X_test = pd.read_csv("review_analysis/data/processed/X_test.csv")
y_test = pd.read_csv("review_analysis/data/processed/y_test.csv")
ls = os.listdir(r"review_analysis/models")

# st.markdown("# Main page")
st.sidebar.markdown("# Dashboard")
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
    
