# source: https://medium.com/nlpgurukool/introduction-to-machine-learning-breast-cancer-case-study-cc21367e2eb8

import pandas as pd
import numpy as np
from sklearn import datasets

# load data
cancer = datasets.load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
print(df.head()) # see cols

# data pre-processing
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# feature engineering
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)

################ MODEL 1 Binary classification with SVM
from sklearn import svm
clf = svm.SVC(gamma = 0.001, C = 100)

model = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
predictions[0:10] # check predictions

# evaluate model 1
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))

################ MODEL 2 Binary classification with RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, random_state = 42)

rf_model = rf.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# evaluate model 2
print(confusion_matrix(y_test, rf_predictions))  
print(classification_report(y_test, rf_predictions))  
print(accuracy_score(y_test, rf_predictions))


################ SAVE MODEL OUTPUT FOR FUTURE RE-USE
import pickle

# to variable
s = pickle.dumps(model)
model2 = pickle.loads(s)

# to file
#pickle.dump(model, open('Other\\svm.pkl', 'wb'))
#model2 = pickle.load(open('Other\\svm.pkl', 'rb'))

model2.predict(X_test)
print('This is the saved model')
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))