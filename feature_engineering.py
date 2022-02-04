from ast import literal_eval
import pandas as pd

def y_to_binary(df, col, threshold):
    df.loc[df[col] < threshold, col] = 0
    df.loc[df[col] >= threshold, col] = 1
    df.rename(columns={col: f'viral'}, inplace = True)

def X_to_drop(df, col):
    return df.drop([col], axis = 1, inplace = True)

def X_to_dummy(df, col):
    df[col] = df[col].fillna(0)
    df.loc[df[col] != 0, col] = 1
    
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
    df[col] = df[col].apply(literal_eval).apply(len)
    df.rename(columns={col: f'{col}_count'}, inplace = True)
    
def X_to_categorical(df, col):
    df[col] = df[col].astype('category')

if __name__ == "__main__":
    
    tweet_df = pd.read_csv(r'C:\Users\loren\Documents\GitHub\intro2ml_2021_final_project\Data\dataset_full.csv', encoding = 'utf-8')
    sent_df = pd.read_csv(r'C:\Users\loren\Documents\GitHub\intro2ml_2021_final_project\Data\sentiment_full.csv', encoding = 'utf-8')
    
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
    
    # turn NaN to zeros
    X_to_dummy(tweets_df, 'quote_url')
    
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

    # tranforms dtypes of categorical/dummy variables
    X_to_categorical(tweets_df, 'quote_url')
    X_to_categorical(tweets_df, 'video')
    X_to_categorical(tweets_df, 'viral')
        
    # dump processed file 
    tweets_df.to_csv(r'C:\Users\loren\Documents\GitHub\intro2ml_2021_final_project\Data\full_sample_processed.csv', encoding = 'utf-8', index = False)
    print(tweets_df.dtypes)    
