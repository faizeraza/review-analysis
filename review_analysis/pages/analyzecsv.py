import streamlit as st
import pandas as pd
from textblob import TextBlob
from collections import Counter
import emoji

#Sentminet analysis
def emoji_helper(review):
    emojis = []
    emojis.extend([c for c in review if emoji.demojize(c) != c]) 
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis)))).rename(columns={0:"Emoji", 1:"Frequency"})
    return emoji_df


st.markdown("Analyze multiple reviews at once!!!")
st.sidebar.markdown("# Bulk Analyzer")
st.sidebar.markdown("""the result csv file will have two updated columns in it :                                  
                     1. Score                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                     2. Analysis .""")
st.header('Sentiment Analysis')
upl = st.file_uploader('Upload file')
st.markdown("Note: the csv file must contain review column as 'reviews'. ")

def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

if upl:
    df = pd.read_csv(upl)
    del df['Unnamed: 0']
    df['score'] = df['reviews'].apply(score)
    df['analysis'] = df['score'].apply(analyze)
    st.write(df.head(10))

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='sentiment.csv',
        mime='text/csv')
    
