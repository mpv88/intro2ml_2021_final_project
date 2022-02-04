from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) # 70% training and 30% test


# D) define and fit/predict models
RF = RandomForestClassifier(bootstrap = True, n_estimators = 250, 
                            class_weight = None, criterion = 'gini',
                            max_depth = None, max_features = 'auto', 
                            max_leaf_nodes = None, min_impurity_decrease = 0.0,
                            min_samples_leaf = 1, min_samples_split = 2, 
                            min_weight_fraction_leaf = 0.0,  n_jobs = 1,
                            oob_score = True, random_state = None, 
                            verbose = 0, warm_start = False)

RF_model = make_pipeline(preprocessor, RF)

model_output = RF_model.fit(X_train, y_train)
y_pred = RF_model.predict(X_test)


# E) cross validation
cv_results = cross_validate(RF_model, X, y, cv = 5)
scores = cv_results['test_score']
print(cv_results)
print('The mean cross-validation accuracy is: 'f'{scores.mean():.3f} +/- {scores.std():.3f}')


# F) evaluate accuracy
print('Confusion matrix', metrics.confusion_matrix(y_test, y_pred))
print('Accuracy out of sample is: ', metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(RF_model.score(X_train, y_train))
print(RF_model.score(X_test, y_test))

# OBB error
oob_error = 1 - RF.oob_score_
print('The OOB error is: ', oob_error)

# print ROC
rfc_disp = metrics.RocCurveDisplay.from_estimator(RF_model, X_test, y_test, alpha=0.8)
plt.show()


# G) evaluate variable importance & plot
feature_imp = pd.Series(RF.feature_importances_).sort_values(ascending = False) #index = list(X) #FIXME
print(feature_imp)

# plot
sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.legend()
plt.show()


# H) save fitted model output for quick future loading
# to variable
#s = pickle.dumps(model_output)
#model2 = pickle.loads(s)

# to file
pickle.dump(model_output, open('intro2ml_2021_final_project\\Data\\rf.pkl', 'wb'))
#model_2 = pickle.load(open('intro2ml_2021_final_project\\Data\\rf.pkl', 'rb'))