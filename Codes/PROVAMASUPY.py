import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
df = pd.read_csv(r'C:\Users\loren\Documents\GitHub\intro2ml_2021_final_project\Data\all_tweets_viral.csv')
print(df)
texts = df["tweet"]
print(texts)
type(texts)
es = texts[0]
print(es)
type(es)
len(es)
lunghezze = [0 for x in texts]
for i in range(len(texts)):
    lunghezze[i]=len(texts[i])
rt = df["retweets_count"]
retweets = [np.int64(0) for x in rt]
for i in range(len(rt)):
    retweets[i] = rt[i]
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
lung2D = np.zeros((len(lunghezze),2),dtype=np.int32)
for i in range(len(lunghezze)):
    lung2D[i][0]=lunghezze[i]
print(min(lunghezze))
print(max(lunghezze))
print(min(retweets))
print(max(retweets))
regr = RandomForestRegressor(n_estimators=250,max_depth=20, random_state=0)
regr.fit(lung2D,retweets)
predpoints = np.zeros((len(lunghezze),1),dtype=float)
for i in range(len(lunghezze)):
    predpoints[i] = regr.predict([[lunghezze[i],0]]) 
    if(i % 1000==1):
        print(i)      
mpl.scatter(lunghezze,predpoints)