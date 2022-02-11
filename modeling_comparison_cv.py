from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


# A) load dataset & define variables
tweets_df = pd.read_csv(r'C:\Users\loren\Documents\GitHub\intro2ml_2021_final_project\Data\full_sample_processed.csv', encoding='utf-8')
tweets_df['weekday'] = tweets_df['weekday'].astype('category')
tweets_df['hour'] = tweets_df['hour'].astype('category')
tweets_df['quote_url'] = tweets_df['quote_url'].astype('category')
tweets_df['video'] = tweets_df['video'].astype('category')
tweets_df['viral'] = tweets_df['viral'].astype('category')

y = tweets_df['viral']  # labels
X = tweets_df.drop(columns=['viral']) # features


# B) setup feature engineering 
categorical_cols_selector = make_column_selector(dtype_include='category')
numerical_cols_selector = make_column_selector(dtype_exclude='category')

categorical_cols = categorical_cols_selector(X)
numerical_cols = numerical_cols_selector(X)

categorical_preprocessor = OneHotEncoder(drop='first', handle_unknown ='ignore')
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([('one-hot-encoder', categorical_preprocessor, categorical_cols),
                                  ('standard-scaler', numerical_preprocessor, numerical_cols)])


# C) split dataset into train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109) # 70% training and 30% test


# D) define models
models = {}

models['Logistic Regression'] = make_pipeline(preprocessor, LogisticRegression()) # Logistic Regression
models['Support Vector Machines'] = make_pipeline(preprocessor, LinearSVC()) # Support Vector Machines
models['Decision Trees'] = make_pipeline(preprocessor, DecisionTreeClassifier()) # Decision Trees
models['Random Forest'] = make_pipeline(preprocessor, RandomForestClassifier()) # Random Forest
models['Naive Bayes'] = make_pipeline(preprocessor, GaussianNB()) # Naive Bayes
models['K-Nearest Neighbor'] = make_pipeline(preprocessor, KNeighborsClassifier()) # K-Nearest Neighbors


# E)  fit/predict models & collect metrcs
accuracy, precision, recall, f1 = {}, {}, {}, {}

#Scores
scoring = {'accuracy' : metrics.make_scorer(metrics.accuracy_score), 
           'precision' : metrics.make_scorer(metrics.precision_score),
           'recall' : metrics.make_scorer(metrics.recall_score), 
           'f1_score' : metrics.make_scorer(metrics.f1_score)}

for key in models.keys():
    scores = cross_validate(models[key], X_train, y_train, cv=5, scoring=scoring)

    accuracy[key] = scores['test_accuracy'].mean()
    print("Accuracy (Testing):  %.3f (+/- %.3f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    precision[key] = scores['test_precision'].mean()
    print("Precision (Testing):  %.3f (+/- %.3f)" % (scores['test_precision'].mean(), scores['test_precision'].std()))
    recall[key] = scores['test_recall'].mean()
    print("Recall (Testing):  %.3f (+/- %.3f)" % (scores['test_recall'].mean(), scores['test_recall'].std()))
    f1[key] = scores['test_f1_score'].mean()
    print("F1-Score (Testing):  %.3f (+/- %.3f)" % (scores['test_f1_score'].mean(), scores['test_f1_score'].std()))

# F)  plot results & compare performances 
# 1. gather table of results
df_model = pd.DataFrame(index=models.keys(), columns=['Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'])
df_model['Mean Accuracy'] = accuracy.values()
df_model['Mean Precision'] = precision.values()
df_model['Mean Recall'] = recall.values()
df_model['Mean F1 Score'] = f1.values()
print(df_model)

# 2. plot metrics comparison
ax  = df_model.plot.bar(rot=45)
ax.legend(ncol = len(models.keys()), loc=8, bbox_to_anchor=(0.25, -0.4, 0.5, 0.5), prop={'size': 14})
plt.title('Evaluation metrics comparison', fontweight='bold', fontsize=30)
plt.tight_layout()
plt.show()

# 3. plot ROC-AUC comparison
''' ax = plt.gca()
plt.title('ROC Curve comparison', fontweight='bold', fontsize=15)
for key in models.keys():
    
    
    
    
    metrics.RocCurveDisplay.from_estimator(models[key], X_train, y_train, name=key, ax=ax, alpha=0.8)
plt.show() '''


# 4. plot ROC-AUC comparison
''' ax = plt.gca()
plt.title('Precision-Recall Curve comparison', fontweight='bold', fontsize=15)
for key in models.keys():
    metrics.PrecisionRecallDisplay.from_estimator(models[key], X_train, y_train, name=key, ax=ax, alpha=0.8)
plt.show() '''