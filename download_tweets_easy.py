# https://github.com/twintproject/twint
# pip3 install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
# NOTE: delete the hashtag in line 92 of url.py

import twint
import pandas as pd


def read_txt(txt_file):
    ''' reads a .txt file returning a list of strings'''
    lst = []
    with open(txt_file) as file:
        while True:
            line = file.readline()
            if not line:
                break
            lst.append(line.strip())
    return lst

#---------------------------------------------------------------------------------
# set up twint configuration
c = twint.Config()

# set up search parameters https://github.com/twintproject/twint/wiki/Configuration
c.Limit = 20 # max n° of tweets
c.Since = '2010-01-01'
c.Until =  '2021-12-25'
#c.Filter_retweets = True # filters out retweets
c.Min_retweets = 1e3 # min n° of retweets
#c.Min_likes = 1e5
#c.Min_replies = 1e4
#c.Popular_tweets = True
c.Lang = 'en'
#c.Lowercase = True
#c.Custom['tweet'] = ['username','created_at','place','geo','near','retweet','retweet_date','replies_count','retweets_count','likes_count','tweet','urls','hashtags']

# available fields for downloading
'''
'cashtags','conversation_id','created_at','datestamp','geo','hashtags','id','likes_count','link','mentions','name',
'near','photos','place','quote_url','replies_count','retweet','retweet_date','retweets_count','source','timestamp',
'timezone','tweet','urls','user_id','user_rt_id','username','video'
'''

# dumping data
c.Count = True
c.Store_csv = True
c.Output = 'intro2ml_2021_final_project\\Data\\all_tweets.csv'
#c.Pandas = True
#---------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # read food list
    food_vocabulary = read_txt('intro2ml_2021_final_project\\food_vocabulary.txt')
    food_search_string = (' OR ').join(food_vocabulary)
    
    # set search string for the engine
    c.Search = 'pizza OR #pizza OR food OR #food' #food_search_string
    
    # run search engine
    twint.run.Search(c)
    '''
    tweets_df = pd.read_csv('intro2ml_2021_final_project\\Data\\all_tweets.csv', names=["id", "created_at", "country", "city", "text", "lang",
                                        "hashtags", "urls", "user_mentions", 
                                        "hashtag_count", "url_count", "mention_count",
                                        "possibly_sensitive", 
                                        "favorite_count", "favorited", "retweet_count", "retweeted",
                                        "user_statuses_count", "user_favorites_count",
                                        "user_follower_count", "user_description", "user_location", "user_timezone"])
    tweets_df.head(10)
'''