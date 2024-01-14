import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import nltk
import numpy as np
from nltk.corpus import stopwords 
import string
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def tokenization(targetColumn):
    df= np.asarray(targetColumn)
    df_token=[]
    for i in df:
        # print(i)
        i= i.lower()  #convert  to lower case
        i = i.translate(str.maketrans('','',string.punctuation))   #removed punctuation 
        df_token.append(nltk.word_tokenize(i)) 
    print("tokenaization Done!")
    return df_token

def removeStopWords(df_token):
    stop_words = list(stopwords.words('english'))
    df_token_filter = []

    lemmatizer = WordNetLemmatizer()

    for i in df_token:
        i = [x for x in i if x not in stop_words]
        i = [lemmatizer.lemmatize(x) for x in i]
        i = ' '.join(i)
        df_token_filter.append(i)
    print("Stop Words Removed Succesfully")
    return df_token_filter

def main():
    #load data set
    df_alexa = pd.read_csv("review_analysis/data/raw/amazon_alexa_data.csv")
    print("data set loaded...")
    #clean data set
    df_alexa.drop(df_alexa.columns[df_alexa.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
    df_alexa.dropna(how= 'any', axis = 0)
    # print(df_alexa.isnull().sum())
    x_cat = df_alexa[['variation']]
    encoder = OneHotEncoder()
    x_cat = pd.DataFrame(encoder.fit_transform(x_cat).toarray())

    # df2  = [df_alexa['verified_reviews'], df_alexa['feedback'], x_cat]
    df_alexa= pd.concat([df_alexa,x_cat], axis=1)
    df_alexa.drop(['variation'],axis=1,inplace=True)
    df_alexa['verified_reviews'].dropna()
    #tokenization of reviews
    df_token = tokenization(df_alexa['verified_reviews'])
        
    # stopwords removed
    # lemmatizing words
    df_alexa['verified_reviews'] = removeStopWords(df_token)

    # transform words to vectors
    countv = CountVectorizer()
    reviews_countv = countv.fit_transform(df_alexa['verified_reviews'])
    df_alexa.drop(['verified_reviews'], inplace =True , axis =1)
    df_alexa2 = pd.DataFrame(reviews_countv.toarray())
    print("Preprocessing Done...")
    # determine X and y
    X = df_alexa2
    y = df_alexa['feedback']

    # split test, train dataset
    X_train,X_test, y_train , y_test = tts(X , y , test_size = 0.2)

    X_train.to_csv("review_analysis/data/processed/X_train.csv")
    X_test.to_csv("review_analysis/data/processed/X_test.csv")
    y_train.to_csv("review_analysis/data/processed/y_train.csv")
    y_test.to_csv("review_analysis/data/processed/y_test.csv")
    print("Data Cleaned...")

if __name__ == "__main__":
    main()

    




