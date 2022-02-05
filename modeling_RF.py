from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
RF = RandomForestClassifier(n_estimators = 100,
                            criterion = 'gini',
                            max_depth = None,
                            min_samples_split = 2,
                            min_samples_leaf = 1,
                            min_weight_fraction_leaf = 0.0,
                            max_features = 'auto',
                            max_leaf_nodes = None,
                            min_impurity_decrease = 0.0,
                            bootstrap = True, 
                            oob_score = False, # default = False
                            n_jobs = None,
                            random_state = None,
                            verbose = 0,
                            warm_start = False,
                            class_weight = None,
                            ccp_alpha = 0.0,
                            max_samples = None)

RF_model = make_pipeline(preprocessor, RF)
model_output = RF_model.fit(X_train, y_train)
y_pred = RF_model.predict(X_test)


# E) cross validation
cv_results = cross_validate(RF_model, X, y, cv = 5)
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
print(RF_model.score(X_train, y_train))
print(RF_model.score(X_test, y_test))

# OBB error
#oob_error = 1 - RF.oob_score_
#print('The OOB error is: ', oob_error)

# print ROC
rfc_disp = metrics.RocCurveDisplay.from_estimator(RF_model, X_test, y_test, alpha=0.8)
plt.show()


# G) evaluate variable importance & plot
labels = [s.replace('one-hot-encoder__','').replace('standard-scaler__','') for s in RF_model[:-1].get_feature_names_out()] # RF_model[1].feature_names_in_
feature_importance = pd.Series(RF.feature_importances_, index = labels).sort_values(ascending = False)
print(feature_importance)

# plot
sns.barplot(x = feature_importance, y = feature_importance.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.legend()
plt.show()


# H) main RF parameters' tuning via random grid search
n_estimators = np.arange(start = 100, stop = 251, step = 50, dtype = int)
criterion = ['gini', 'entropy']
max_depth = np.arange(start = 1, stop = 201, step = 25, dtype = int)
min_samples_split = np.arange(start = 1, stop = 31, step = 5, dtype = int)
min_samples_leaf = np.arange(start = 1, stop = 5, step = 1, dtype = int)
max_features = ['auto', 'sqrt']
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_features': max_features,
               'bootstrap': bootstrap,
               }

RF_random = RandomizedSearchCV(estimator = RF,
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

# fit/predict new optimised RF classifier
RF_random_model = make_pipeline(preprocessor, RF_random)
model_random_output = RF_random_model.fit(X_train, y_train)
y_pred_random = RF_random_model.predict(X_test)

# check which are the chosen params 
print(RF_random.best_params_)

# evaluate updated metrics
print(RF_random_model.score(X_train, y_train))
print(RF_random_model.score(X_test, y_test))


# I) main RF parameters' refinement via grid search
search_grid = {'n_estimators': np.linspace(start = 150, stop = 300, num = 5, dtype = int),
                'min_samples_split': np.linspace(start = 10, stop = 20, num = 5, dtype = int),
                'min_samples_leaf': np.linspace(start = 2, stop = 6, num = 4, dtype = int),
                'max_depth': np.linspace(start = 140, stop = 180, num = 4, dtype = int)}

RF_refined = RandomForestClassifier(n_estimators = 100,
                                    criterion = 'gini',
                                    max_depth = None,
                                    min_samples_split = 2,
                                    min_samples_leaf = 1,
                                    min_weight_fraction_leaf = 0.0,
                                    max_features = 'sqrt',
                                    max_leaf_nodes = None,
                                    min_impurity_decrease = 0.0,
                                    bootstrap = True, 
                                    oob_score = True, # default = True
                                    n_jobs = None,
                                    random_state = None,
                                    verbose = 0,
                                    warm_start = False,
                                    class_weight = None,
                                    ccp_alpha = 0.0,
                                    max_samples = None)

RF_search = GridSearchCV(estimator = RF_refined,
                         param_grid = search_grid,
                         scoring = None,
                         n_jobs = None,
                         refit = True,
                         cv = None, # None = 5 folds
                         verbose = 0,
                         pre_dispatch = '2*n_jobs',
                         error_score = np.nan,
                         return_train_score = False)

# fit/predict new optimised RF classifier
RF_search_model = make_pipeline(preprocessor, RF_search)
model_search_output = RF_search_model.fit(X_train, y_train)
y_pred_search = RF_search_model.predict(X_test)

# check which are the chosen params 
print(RF_search.best_params_)

# evaluate updated metrics
print(RF_search_model.score(X_train, y_train))
print(RF_search_model.score(X_test, y_test))


# J) save fitted model output for quick future loading
# to variable
#s = pickle.dumps(model_output)
#model2 = pickle.loads(s)

# to file
pickle.dump(model_output, open('intro2ml_2021_final_project\\Data\\rf_full.pkl', 'wb'))
#model_2 = pickle.load(open('intro2ml_2021_final_project\\Data\\rf.pkl', 'rb'))