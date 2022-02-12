from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# A) load dataset & define variables
tweets_df = pd.read_csv('intro2ml_2021_final_project\\Data\\2k_sample_processed.csv', encoding='utf-8')
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

X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
X_test = pd.DataFrame(preprocessor.fit_transform(X_test))

# D) define models
models = {}

models['Logistic Regression'] = LogisticRegression() # Logistic Regression
models['Support Vector Machines'] = LinearSVC() # Support Vector Machines
models['Decision Trees'] = DecisionTreeClassifier() # Decision Trees
models['Random Forest'] = RandomForestClassifier() # Random Forest
models['Naive Bayes'] = GaussianNB() # Naive Bayes
models['K-Nearest Neighbor'] = KNeighborsClassifier() # K-Nearest Neighbors


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
    print('{} average Accuracy (Validation):  %.3f (+/- %.3f)'.format(key) % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    precision[key] = scores['test_precision'].mean()
    print('{} average Precision (Validation):  %.3f (+/- %.3f)'.format(key) % (scores['test_precision'].mean(), scores['test_precision'].std()))
    recall[key] = scores['test_recall'].mean()
    print('{} average Recall (Validation):  %.3f (+/- %.3f)'.format(key) % (scores['test_recall'].mean(), scores['test_recall'].std()))
    f1[key] = scores['test_f1_score'].mean()
    print('{} average F1-Score (Validation):  %.3f (+/- %.3f)'.format(key) % (scores['test_f1_score'].mean(), scores['test_f1_score'].std()))

# F)  plot results & compare performances 
# 1. gather table of results
df_model = pd.DataFrame(index=models.keys(), columns=['Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'])
df_model['Mean Accuracy'] = accuracy.values()
df_model['Mean Precision'] = precision.values()
df_model['Mean Recall'] = recall.values()
df_model['Mean F1 Score'] = f1.values()
print(df_model)

# 2. plot metrics comparison
ax = df_model.plot.bar(rot=0)
ax.legend(ncol = len(models.keys()), loc=8, bbox_to_anchor=(0.25, -0.1, 0.5, 0.5), prop={'size': 12})
plt.title('Performance metrics comparison for 5-fold CV', fontweight='bold', fontsize=15)
plt.show()

# 3. plot ROC-AUC comparison
cv = StratifiedKFold(n_splits=5 , shuffle=False)
ax = plt.gca()

for key in models.keys():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        models[key].fit(X_train.iloc[train], y_train.iloc[train])
        viz =  metrics.RocCurveDisplay.from_estimator(models[key], X_train.iloc[test], y_train.iloc[test], name='_nolegend_', alpha=0.0, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, label=r"Mean ROC {} (AUC = %0.2f $\pm$ %0.2f)".format(key) % (mean_auc, std_auc), lw=2, alpha=0.8)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
plt.title('Mean ROC Curves comparison for 5-fold CV', fontweight='bold', fontsize=15)
ax.legend(loc="lower right")
plt.show()


# 4. plot ROC-AUC comparison
ax = plt.gca()
plt.title('Mean Precision-Recall Curves comparison for 5-fold CV', fontweight='bold', fontsize=15)
for key in models.keys():
    metrics.PrecisionRecallDisplay.from_estimator(models[key], X_train, y_train, name=key, ax=ax, alpha=0.8)
plt.show()