import streamlit as st
from textblob import TextBlob
import cleantext
import nltk
from nltk.corpus import wordnet

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
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))


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
