import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def y_to_binary(df, col, threshold):
    df.loc[df[col] < threshold, col] = 0
    df.loc[df[col] >= threshold, col] = 1
    return df

def X_to_drop(df, col):
    return df.drop([col], axis = 1, inplace = True)
    
def X_to_length(df, col):
    df[col] = df[col].astype(str).str.len()
    df.rename(columns={col: f'{col}_length'}, inplace = True)

def X_to_weekday(df, col):
    df[col] = pd.to_datetime(df[col]).dt.day_name().astype('category')
    df.rename(columns={col: f'weekday'}, inplace = True)

def X_to_hour(df, col):
    df[col] = pd.to_datetime(df[col]).dt.hour.astype('category')
    df.rename(columns={col: f'hour'}, inplace = True)
    
def X_to_count_list(df, col):
    df[col] =  df[col].str.len()
    df.rename(columns={col: f'{col}_count'}, inplace = True)
#FIXME

if __name__ == "__main__":
    
    tweet_df = pd.read_csv('intro2ml_2021_final_project\\Data\\dataset_2k.csv', encoding = 'utf-8')
    sent_df = pd.read_csv('intro2ml_2021_final_project\\Data\\sentiment_by_sentence.csv', encoding = 'utf-8')
    
    # join 2 datasets
    tweets_df = pd.concat([tweet_df, sent_df], axis=1)
    
    # transforms variable to binary
    y_to_binary(tweets_df, 'retweets_count', threshold = 1000)
    
    # drop variables
    X_to_drop(tweets_df, 'replies_count')
    X_to_drop(tweets_df, 'likes_count')
    X_to_drop(tweets_df, 'id')
    X_to_drop(tweets_df, 'language')
    X_to_drop(tweets_df, 'retweet')
    
    # transforms variable to length
    X_to_length(tweets_df, 'tweet')
    
     # transforms date to weekday
    X_to_weekday(tweets_df, 'date')

    # transforms time to hour bins
    X_to_hour(tweets_df, 'time')

    # transforms list variables to count of their elements
    X_to_count_list(tweets_df, 'mentions')
    X_to_count_list(tweets_df, 'urls')
    X_to_count_list(tweets_df, 'photos')
    X_to_count_list(tweets_df, 'hashtags')
    '''
    # encode categorical/dummy variables
    oht_enc = OneHotEncoder()
    tweets_df['weekday'] = ord_enc.fit_transform(tweets_df['weekday'])
    tweets_df['hour'] = ord_enc.fit_transform(tweets_df['hour'])
    tweets_df['video'] = ord_enc.fit_transform(tweets_df['video'])
    tweets_df['quote_url'] = ord_enc.fit_transform(tweets_df['quote_url'])
    
    # standardize numerical variables
    sc = StandardScaler()
    normed_train_data = pd.DataFrame(sc.fit_transform(training), columns = X.columns)
    '''
    # dump processed file 
    tweets_df.to_csv('intro2ml_2021_final_project\\Data\\2k_sample_processed.csv', encoding = 'utf-8', index = False)
    print(tweets_df.head(10))    
