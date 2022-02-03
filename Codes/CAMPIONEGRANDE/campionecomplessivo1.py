import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
	df = pd.read_csv(r'C:\Users\loren\Documents\GitHub\intro2ml_2021_final_project\Data\dataset_full.csv')
	texts = df["tweet"]
	lunghezze = [0 for x in texts]
	for i in range(len(texts)):
    	lunghezze[i]=len(texts[i])
    rt = df["retweets_count"]
	retweets = [np.int64(0) for x in rt]
	for i in range(len(rt)):
    	retweets[i] = rt[i]
    popular = np.zeros((len(lunghezze),1),dtype=np.int32)
	for i in range(len(lunghezze)):
    	if retweets[i]>=1000:
        	popular[i]=1
	photos = df["photos"]
	numphoto = np.zeros((len(photos),1),dtype=np.int32)
	for i in range(len(photos)):
    	if(photos[i]!=photos[0]):
        	numphoto[i]=len(photos[i].split(','))
    ht = df["hashtags"]
    nhash = np.zeros((len(ht),1),dtype=np.int32)
	for i in range(len(ht)):
    	if(ht[i]!=ht[0]):
        	nhash[i]=len(ht[i].split(','))
	sf = pd.read_csv(r'C:\Users\loren\Documents\GitHub\intro2ml_2021_final_project\Data\sentiment_by_sentence.csv')
	X = np.zeros((len(lunghezze),14),dtype=np.int32)
	for i in range(len(lunghezze)):
    	X[i][0]=lunghezze[i]
    	X[i][1]=nhash[i]
    	X[i][2]=numphoto[i]
    	X[i][3]=nvideo[i]
    	X[i][4]=sf["anger"][i]
    	X[i][5]=sf["anticipation"][i]
    	X[i][6]=sf["disgust"][i]
    	X[i][7]=sf["fear"][i]
    	X[i][8]=sf["joy"][i]
    	X[i][9]=sf["sadness"][i]
    	X[i][10]=sf["surprise"][i]
    	X[i][11]=sf["trust"][i]
    	X[i][12]=sf["negative"][i]
    	X[i][13]=sf["positive"][i]

    # TEST RANDOM FOREST

    y=popular.ravel()
    scores = cross_val_score(regr, X, y, cv=100)
    print(scores.mean())
	print(scores.std())


