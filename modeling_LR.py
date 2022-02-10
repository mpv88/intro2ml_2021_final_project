from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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
LR = LogisticRegression(penalty = 'l2', # {'l1', 'l2', 'elasticnet', 'none'}, l1=lasso l2=ridge
                        dual = False,
                        tol = 1e-4,
                        C = 1.0,
                        fit_intercept = True,
                        intercept_scaling = 1,
                        class_weight = None, # None = all classes supposed to have weight one
                        random_state = None, # used for solver = ('sag', 'saga','liblinear')
                        solver = 'lbfgs', # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
                        max_iter = 100,
                        multi_class = 'auto', # {'auto', 'ovr', 'multinomial'}
                        verbose = 0,
                        warm_start = False,
                        n_jobs = None,
                        l1_ratio = None)

LR_model = make_pipeline(preprocessor, LR)
model_output = LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)


# E) cross validation
cv_results = cross_validate(LR_model, X_train, y_train, cv = 5)
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
print(LR_model.score(X_train, y_train))
print(LR_model.score(X_test, y_test))

# print ROC
rfc_disp = metrics.RocCurveDisplay.from_estimator(LR_model, X_test, y_test, alpha=0.8)
plt.show()


# G) evaluate variable importance & plot
labels = [s.replace('one-hot-encoder__','').replace('standard-scaler__','') for s in LR_model[:-1].get_feature_names_out()]
feature_importance = pd.Series(np.transpose(abs(LR.coef_).tolist()[0]), index=labels).sort_values(ascending = False)
print(feature_importance)

# plot
sns.barplot(x = feature_importance, y = feature_importance.index)
plt.xlabel('Most constributing features by coefficient')
plt.ylabel('Features')
plt.title('LR: Importance of Features')
plt.legend()
plt.show()


# H) main LR parameters' tuning via random grid search
penalty = ['none', 'l1', 'l2', 'elasticnet']
C = [100, 10, 1.0, 0.1, 0.01]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

random_grid = {'penalty': penalty,
               'C': C,
               'solver': solver,
               }

LR_random = RandomizedSearchCV(estimator = LR,
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

# fit/predict new optimised LR classifier
LR_random_model = make_pipeline(preprocessor, LR_random)
model_random_output = LR_random_model.fit(X_train, y_train)
y_pred_random = LR_random_model.predict(X_test)

# check which are the chosen params 
print(LR_random.best_params_)
print(LR_random.best_estimator_)

# evaluate updated metrics
print(LR_random_model.score(X_train, y_train))
print(LR_random_model.score(X_test, y_test))


# I) main LR parameters' refinement via grid search
search_grid = {'C': np.linspace(start = 0.005, stop = 0.05, num = 10, dtype = int),
                }

LR_refined = LogisticRegression(penalty = 'none', # {'l1', 'l2', 'elasticnet', 'none'}
                                dual = False,
                                tol = 1e-4,
                                C = 1.0,
                                fit_intercept = True,
                                intercept_scaling = 1,
                                class_weight = None, # None = all classes supposed to have weight one
                                random_state = None, # used for solver = ('sag', 'saga','liblinear')
                                solver = 'newton-cg', # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
                                max_iter = 100,
                                multi_class = 'auto', # {'auto', 'ovr', 'multinomial'}
                                verbose = 0,
                                warm_start = False,
                                n_jobs = None,
                                l1_ratio = None)

LR_search = GridSearchCV(estimator = LR_refined,
                         param_grid = search_grid,
                         scoring = None,
                         n_jobs = None,
                         refit = True,
                         cv = None, # None = 5 folds
                         verbose = 0,
                         pre_dispatch = '2*n_jobs',
                         error_score = np.nan,
                         return_train_score = False)

# fit/predict new optimised LR classifier
LR_search_model = make_pipeline(preprocessor, LR_search)
model_search_output = LR_search_model.fit(X_train, y_train)
y_pred_search = LR_search_model.predict(X_test)

# check which are the chosen params 
print(LR_search.best_params_)
print(LR_search.best_estimator_)

# evaluate updated metrics
print(LR_search_model.score(X_train, y_train))
print(LR_search_model.score(X_test, y_test))


# J) save fitted model output for quick future loading
# to variable
#s = pickle.dumps(model_output)
#model2 = pickle.loads(s)

# to file
pickle.dump(model_output, open('intro2ml_2021_final_project\\Data\\lr_full.pkl', 'wb'))
#model_2 = pickle.load(open('intro2ml_2021_final_project\\Data\\lr_full.pkl', 'rb'))