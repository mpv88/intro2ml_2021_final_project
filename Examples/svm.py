from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# a) load dataset
cancer = datasets.load_breast_cancer()

# print the y labels (0:malignant, 1:benign) and X features
print("Labels: ", cancer.target_names)
print(cancer.target)
print("Features: ", cancer.feature_names)
cancer.data.shape
print(cancer.data[0:5])

# b) split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.3, random_state = 109) # 70% training and 30% test


# c) train/predict a SVM linear classifier
clf = svm.SVC(kernel = 'linear')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# d) evaluate accuracy
print("Confusion matrix", metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))