from sklearn.naive_bayes import MultinomialNB
import pandas as pd
# from sklearn.metrics import accuracy_score as ac
# from sklearn.metrics import confusion_matrix as cm
import pickle

def mnbmodeltrain():
    X_train = pd.read_csv("review_analysis/data/processed/X_train.csv")
    y_train = pd.read_csv("review_analysis/data/processed/y_train.csv")
    print(y_train['feedback'])
    model = MultinomialNB().fit(X_train, y_train['feedback'])
    return model

def saveModel(model,filename):
    pickle.dump(model,open(filename, 'wb'))

if __name__ == "__main__":
    model = mnbmodeltrain()
    modelname = "review_analysis/models/MNBClassifier"
    saveModel(model,modelname)
    # X_test = pd.read_csv("review_analysis/data/processed/X_test.csv")
    # y_test = pd.read_csv("review_analysis/data/processed/y_test.csv")
    # filename = "review_analysis/models/MNBClassifier"
    # loaded_model = pickle.load(open(filename, 'rb'))
    # prediction = loaded_model.predict(X_test)
    # ac_mnb = ac(y_test['feedback'], prediction)
    # print(ac_mnb)
    # print(cm(y_test['feedback'],prediction))