import streamlit as st
from textblob import TextBlob
import cleantext
import nltk
from nltk.corpus import wordnet
from collections import Counter
import emoji
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.markdown("Analyze text")
st.sidebar.markdown("# analyzer")
st.header('Sentiment Analysis')

def negation_handler(sentence):
    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i-1] in ['not',"n't","'t"]:
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp>max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
    while '' in sentence:
        sentence.remove('')
    for i in range(len(sentence)):
      if "ca" == sentence[i]:
        sentence[i]="can"
    return sentence

#Sentminet analysis
def emoji_helper(review):
    emojis = []
    emojis.extend([c for c in review if emoji.demojize(c) != c]) 
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis)))).rename(columns={0:"Emoji", 1:"Frequency"})
    return emoji_df

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        if score >= 0.5:
            st.write('Positive')
        elif score <= -0.5:
            st.write('Negative')
        else:
            st.write('Neutral')
        st.write('Polarity: ', round(score,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))
        emoji_df = emoji_helper(text)
        st.write("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()

            ax.pie(emoji_df["Frequency"].head(),labels=emoji_df["Emoji"].head(),autopct="%0.2f" )
            st.pyplot(fig)

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))
        st.sidebar.write("Cleaning:")
        st.sidebar.write('extra space removed...')
        st.sidebar.write('stopwords removed...')
        st.sidebar.write('transformed to lowercase...')
    handel = st.text_input('Handel Negation: ')
    if handel:
        sentence =nltk.word_tokenize(handel)
        sen = " ".join(negation_handler(sentence))
        st.write(sen)
        st.sidebar.write('Negation Handeling:')
        st.sidebar.write('''This Feature aims to handle negations within sentences.
                 It transforms negated words into their antonyms, enhancing the accuracy of sentiment detection and language comprehension.''')
