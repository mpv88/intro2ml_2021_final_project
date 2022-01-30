import pandas as pd


def filter_eng_tw(df):
    return df[(df.language=='en')]

def filter_string_tw(df, word):
    return df[df.tweet != word]

def filter_viral_threshold(df, threshold, over):
    df['retweets_count'] = pd.to_numeric(df['retweets_count'])
    if over == True:
        filtered=df[df.retweets_count >= threshold]
    else:
        filtered=df[df.retweets_count < threshold]
    return filtered

def filter_duplicate_tw(df):
    return df.drop_duplicates(subset='id', keep='first')


if __name__ == "__main__":
    
    tweets_df = pd.read_csv('intro2ml_2021_final_project\\Data\\dataset_full.csv', encoding = 'utf-8')
                            #, names=['id','lang','retweeted','date','time','text','mentions','urls','photos','hashtags','quote_url','video',
                                 #  'replies_count','likes_count','retweet_count']
                                    # "hashtag_count", "url_count", "mention_count"
    tweets_df.head(10)
    
    # remove non-english tweets
    filtered_tweets = filter_eng_tw(tweets_df)
    
    # remove snoopy tweets
    filtered_tweets = filter_string_tw(filtered_tweets, 'Snoopy')
    
    # remove tweets with retweets under/over threshold
    filtered_tweets = filter_viral_threshold(filtered_tweets, threshold=1000, over=False)
    
    # remove duplicate tweets
    filtered_tweets = filter_duplicate_tw(filtered_tweets)
    
    # dump processed file 
    filtered_tweets.to_csv('intro2ml_2021_final_project\\Data\\all_tweets_filtered.csv')
    filtered_tweets.shape
    filtered_tweets.head(10)    