from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# A) load dataset & define variables
tweets_df = pd.read_csv('intro2ml_2021_final_project\\Data\\2k_sample_processed.csv', encoding = 'utf-8')
tweets_df['weekday'] = tweets_df['weekday'].astype('category')
tweets_df['hour'] = tweets_df['hour'].astype('category')
tweets_df['quote_url'] = tweets_df['quote_url'].astype('category')
tweets_df['video'] = tweets_df['video'].astype('category')
tweets_df['viral'] = tweets_df['viral'].astype('category')
tweets_df.head()

y = tweets_df['viral']  # labels
X = tweets_df.drop(columns = ['viral']) # features
print(X.info())


# B) setup feature engineering 
categorical_cols_selector = make_column_selector(dtype_include = 'category')
numerical_cols_selector = make_column_selector(dtype_exclude = 'category')

categorical_cols = categorical_cols_selector(X)
print(categorical_cols)
numerical_cols = numerical_cols_selector(X)
print(numerical_cols)

categorical_preprocessor = OneHotEncoder(drop = 'first', handle_unknown = "ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([('one-hot-encoder', categorical_preprocessor, categorical_cols),
                                  ('standard-scaler', numerical_preprocessor, numerical_cols)])


# C) split dataset into train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 109) # 70% training and 30% test


# D) define models
models = {}

models['Logistic Regression'] = make_pipeline(preprocessor, LogisticRegression()) # Logistic Regression
models['Support Vector Machines'] = make_pipeline(preprocessor, LinearSVC()) # Support Vector Machines
models['Decision Trees'] = make_pipeline(preprocessor, DecisionTreeClassifier()) # Decision Trees
models['Random Forest'] = make_pipeline(preprocessor, RandomForestClassifier()) # Random Forest
models['Naive Bayes'] = make_pipeline(preprocessor, GaussianNB()) # Naive Bayes
models['K-Nearest Neighbor'] = make_pipeline(preprocessor, KNeighborsClassifier()) # K-Nearest Neighbors


# E)  fit/predict models & collect metrcs
accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    models[key].fit(X_train, y_train) # model fit
    y_pred = models[key].predict(X_test)  # model predict
    
    accuracy[key] = metrics.accuracy_score(y_test, y_pred)
    precision[key] = metrics.precision_score(y_test, y_pred)
    recall[key] = metrics.recall_score(y_test, y_pred)


# F)  plot results & compare performances 
# gather table of results
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
print(df_model)

# plot metrics
ax  = df_model.plot.bar(rot=45)
ax.legend(ncol= len(models.keys()), bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 14})
plt.tight_layout()
plt.show()