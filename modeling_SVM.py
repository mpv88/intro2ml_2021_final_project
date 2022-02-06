from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


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


# D) define and fit/predict models
SVM = SVC(C = 1.0, # regularization parameter
          kernel = 'rbf', # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
          degree = 3, # polynomial degree (poly only, ignored by others)
          gamma = 'scale', # kernel coefficient (rbf, poly, sigmoid only)
          coef0 = 0.0, # independent term of kernel (poly, sigmoid only)
          shrinking = True,
          probability = False,
          tol = 1e-3,
          cache_size = 200,
          class_weight = None, # None = all classes have weight one
          verbose = False,
          max_iter = -1, # -1 = no limit
          decision_function_shape = 'ovr', # ignored for binary classification
          break_ties = False,
          random_state = None)  # ignored when probability=False

SVM_model = make_pipeline(preprocessor, SVM)
model_output = SVM_model.fit(X_train, y_train)
y_pred = SVM_model.predict(X_test)


# E) cross validation
cv_results = cross_validate(SVM_model, X, y, cv = 5)
scores = cv_results['test_score']
print(cv_results)
print('The mean cross-validation accuracy is: 'f'{scores.mean():.3f} +/- {scores.std():.3f}')


# F) evaluate model via stats
print('Confusion matrix', metrics.confusion_matrix(y_test, y_pred))
print('Accuracy out of sample is: ', metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 score:", metrics.f1_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(SVM_model.score(X_train, y_train))
print(SVM_model.score(X_test, y_test))

# print ROC
rfc_disp = metrics.RocCurveDisplay.from_estimator(SVM_model, X_test, y_test, alpha=0.8)
plt.show()


# G) evaluate variable importance & plot JUST FOR LINEAR KERNEL 
# (see https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn) & sklearn.inspection.permutation_importance
'''
labels = [s.replace('one-hot-encoder__','').replace('standard-scaler__','') for s in SVM_model[:-1].get_feature_names_out()]
feature_importance = pd.Series(np.transpose(abs(SVM.coef_).tolist()[0]), index=labels).sort_values(ascending = False)
print(feature_importance)

# plot
sns.barplot(x = feature_importance, y = feature_importance.index)
plt.xlabel('Most constributing features by coefficient')
plt.ylabel('Features')
plt.title('SVM: Importance of Features')
plt.legend()
plt.show()
'''

# H) main SVM parameters' tuning via random grid search
C = [0.1, 1, 10, 100, 1000]
gamma = [1, 0.1, 0.01, 0.001, 0.0001]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
degree = [2, 3, 4]

random_grid = {'C': C,
               'gamma': gamma,
               'kernel': kernel,
               'degree': degree
               }

SVM_random = RandomizedSearchCV(estimator = SVM,
                               param_distributions = random_grid,
                               n_iter = 10, # default = 10
                               scoring = None,
                               n_jobs = None,
                               refit = True,
                               cv = None, # None = 5 folds
                               verbose = 0,
                               pre_dispatch = '2*n_jobs',
                               random_state = 32, # default = None
                               error_score = np.nan,
                               return_train_score = False)

# fit/predict new optimised SVM classifier
SVM_random_model = make_pipeline(preprocessor, SVM_random)
model_random_output = SVM_random_model.fit(X_train, y_train)
y_pred_random = SVM_random_model.predict(X_test)

# check which are the chosen params 
print(SVM_random.best_params_)
print(SVM_random.best_estimator_)

# evaluate updated metrics
print(SVM_random_model.score(X_train, y_train))
print(SVM_random_model.score(X_test, y_test))


# I) main SVM parameters' refinement via grid search
search_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'degree': [2, 3, 4]}

SVM_refined = SVC(C = 1.0, # regularization parameter
                  kernel = 'rbf', # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
                  degree = 3, # polynomial degree (poly only, ignored by others)
                  gamma = 'scale', # kernel coefficient (rbf, poly, sigmoid only)
                  coef0 = 0.0, # independent term of kernel (poly, sigmoid only)
                  shrinking = True,
                  probability = False,
                  tol = 1e-3,
                  cache_size = 200,
                  class_weight = None, # None = all classes have weight one
                  verbose = False,
                  max_iter = -1, # -1 = no limit
                  decision_function_shape = 'ovr', # ignored for binary classification
                  break_ties = False,
                  random_state = None) 

SVM_search = GridSearchCV(estimator = SVM_refined,
                         param_grid = search_grid,
                         scoring = None,
                         n_jobs = None,
                         refit = True,
                         cv = None, # None = 5 folds
                         verbose = 0,
                         pre_dispatch = '2*n_jobs',
                         error_score = np.nan,
                         return_train_score = False)

# fit/predict new optimised SVM classifier
SVM_search_model = make_pipeline(preprocessor, SVM_search)
model_search_output = SVM_search_model.fit(X_train, y_train)
y_pred_search = SVM_search_model.predict(X_test)

# check which are the chosen params 
print(SVM_search.best_params_)
print(SVM_search.best_estimator_)

# evaluate updated metrics
print(SVM_search_model.score(X_train, y_train))
print(SVM_search_model.score(X_test, y_test))


# J) save fitted model output for quick future loading
# to variable
#s = pickle.dumps(model_output)
#model2 = pickle.loads(s)

# to file
pickle.dump(model_output, open('intro2ml_2021_final_project\\Data\\svm_full.pkl', 'wb'))
#model_2 = pickle.load(open('intro2ml_2021_final_project\\Data\\svm_full.pkl', 'rb'))