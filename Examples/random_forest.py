import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# a) load dataset
iris = datasets.load_iris()

# print the iris labels (0:setosa, 1:versicolor, 2:virginica)
print(iris.target)


# b) convert dataset into df and assign X,y
data=pd.DataFrame({'sepal length':iris.data[:,0],
                   'sepal width':iris.data[:,1],
                   'petal length':iris.data[:,2],
                   'petal width':iris.data[:,3],
                   'species':iris.target
})
data.head()

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels


# c) split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) # 70% training and 30% test


# d) train/predict Gaussian Classifier
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                             oob_score=False, random_state=None, verbose=0,
                             warm_start=False)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# e) evaluate accuracy
print("Confusion matrix", metrics.confusion_matrix(y_test, y_pred))
print("Accuracy", metrics.accuracy_score(y_test, y_pred))

# single prediction given inputs
print(clf.predict([[3, 5, 4, 2]]))

# f) evaluate variable importance & plot
feature_imp = pd.Series(clf.feature_importances_, index = iris.feature_names).sort_values(ascending = False)
print(feature_imp)

# bar plot
sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.legend()
plt.show()

# g) retrain the model just on 3 features
X = data[['petal length', 'petal width','sepal length']]  # removed unuseful feature "sepal length"
y = data['species']
                                       
# training/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70, random_state = 5) # 70% train/30% test

# fit/predict RF classifier
clf=RandomForestClassifier(n_estimators = 100, oob_score = True)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

# evaluate accuracy
print("Confusion matrix 2", metrics.confusion_matrix(y_test, y_pred))
print("Accuracy 2", metrics.accuracy_score(y_test, y_pred))
#print (clf.score(X_train , y_train))
#print(clf.score(y_test, y_pred))

# OBB error
oob_error = 1 - clf.oob_score_
print(oob_error)

# print ROC
rfc_disp = metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test, alpha=0.8)
plt.show()