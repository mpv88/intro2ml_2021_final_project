# https://github.com/twintproject/twint
# pip3 install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
# NOTE: delete the hashtag in line 92 of url.py

import twint


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

# search parameters https://github.com/twintproject/twint/wiki/Configuration
c.Limit = 200 # max n° of tweets
c.Since = '2006-03-21' # first tweet ever
c.Until =  '2020-01-01'
#c.Min_retweets = 1e3 # min n° of retweets
#c.Min_likes = 1e5
#c.Min_replies = 1e4
#c.Popular_tweets = True
#c.Filter_retweets = True # filters out retweets
c.Lang = 'en'
c.Lowercase = True
c.Count = True
c.Store_csv = True
c.Output = 'intro2ml_2021_final_project\\Data\\all_tweets.csv'
c.Custom['tweet'] = ['id','language','retweet','date','time','tweet','mentions','urls','photos','hashtags','quote_url','video','replies_count','likes_count','retweets_count']

# available fields for downloading
'''
'cashtags','conversation_id','created_at','datestamp','geo','hashtags','id','likes_count','link','mentions','name',
'near','photos','place','quote_url','replies_count','retweet','retweet_date','retweets_count','source','timestamp',
'timezone','tweet','urls','user_id','user_rt_id','username','video'
'''

def tw_scrap(list):
    for element in list:
        c.Search = element
        twint.run.Search(c)
        print(f'____________________SEARCH FINISHED FOR {element=}_____________________')

#---------------------------------------------------------------------------------


if __name__ == "__main__":
    
    # read food list
    food_vocabulary = read_txt('intro2ml_2021_final_project\\Data\\food_vocabulary.txt')
    
    # scrap food list
    tw_scrap(food_vocabulary)