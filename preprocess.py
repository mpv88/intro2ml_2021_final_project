import pandas as pd

def filter_eng_tw(df):
    return df[(df.language=='en')]

def filter_string_tw(df, word):
    return df[df.tweet != word]

def filter_duplicate_tw(df):
    return df.drop_duplicates(subset='id', keep='first')

def to_binary(df, col, threshold=1000):
    pass
    
def to_count(df, col):
    pass


if __name__ == "__main__":
    
    tweets_df = pd.read_csv('intro2ml_2021_final_project\\Data\\all_tweets.csv')
                            #, names=['id','lang','retweeted','date','time','text','mentions','urls','photos','hashtags','quote_url','video',
                                 #  'replies_count','likes_count','retweet_count']
                                    # "hashtag_count", "url_count", "mention_count"
    tweets_df.head(10)
    
    # remove non-english tweets
    filtered_tweets = filter_eng_tw(tweets_df)
    
    # remove snoopy tweets
    filtered_tweets = filter_string_tw(filtered_tweets, 'Snoopy')
    
    # remove duplicate tweets
    filtered_tweets = filter_duplicate_tw(filtered_tweets)
    
    # dump processed file 
    filtered_tweets.to_csv('intro2ml_2021_final_project\\Data\\all_tweets_filtered.csv')
    filtered_tweets.shape
    
    # transforms variable to binary
    to_binary(tweets_df, variable, threshold=1000)
    
    # transforms variable to count
    to_count(tweets_df, variable)
    