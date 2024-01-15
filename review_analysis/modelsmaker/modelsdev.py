from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import confusion_matrix as cm
import pickle

X_train = pd.read_csv("review_analysis/data/processed/X_train.csv")
y_train = pd.read_csv("review_analysis/data/processed/y_train.csv")
def mnbmoClassifier():
    print(y_train['feedback'])
    model = MultinomialNB().fit(X_train, y_train['feedback'])
    return model
def regression():
    regr = lr()
    regr = regr.fit(X_train,y_train["feedback"])
    return regr

def knnClassifier():
    knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train,y_train["feedback"])
    return knn
def saveModel(model,filename):
    pickle.dump(model,open(filename, 'wb'))

if __name__ == "__main__":
    mnbmodel = mnbmoClassifier()
    modelname = "review_analysis/models/MNBClassifier"
    saveModel(mnbmodel,modelname)
    rgrmodel = regression()
    saveModel(rgrmodel,"review_analysis/models/lregression")
    knn = knnClassifier()
    saveModel(knn,"review_analysis/models/knnClassifier")
    # Bellow code is for test case accuracy_score check
    # X_test = pd.read_csv("review_analysis/data/processed/X_test.csv")
    # y_test = pd.read_csv("review_analysis/data/processed/y_test.csv")
    # ls = ["MNBClassifier","lregression","knnClassifier"]
    # for model in ls:
    #     filename = f"review_analysis/models/{model}"
    #     loaded_model = pickle.load(open(filename, 'rb'))
    #     prediction = loaded_model.predict(X_test)
    #     ac_mnb = ac(y_test['feedback'], prediction)
    #     print(model,ac_mnb)
    #     print(cm(y_test['feedback'],prediction))