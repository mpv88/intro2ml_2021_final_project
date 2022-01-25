# only save information for retweets
import csv
import tweepy as tw
import pandas as pd

# get retweet status
def try_retweet(status, attribute):
    try:
        if getattr(status, attribute):
            return True
    except AttributeError:
        return None

# get country status
def try_country(status, attribute):
    if getattr(status, attribute) != None:
        place = getattr(status, attribute)
        return place.country
    return None

# get city status
def try_city(status, attribute):
    if getattr(status, attribute) != None:
        place = getattr(status, attribute)
        return place.full_name
    return None

# function that tries to get attribute from object
def try_get(status, attribute):
    try:
        return getattr(status, attribute).encode('utf-8')
    except AttributeError:
        return None

# open & write .csv file
csvFile = open('smallsample.csv', 'a')
csvWriter = csv.writer(csvFile)
    
class MyListener(tw.Stream):

    def on_data(self, data):
        try:
            with open('python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
                print("Error on_data: %s" % str(e))
        return True

    def on_status(self, status):
        try:
            # if this represents a retweet
            if try_retweet(status,'retweeted_status'):
                status = status.retweeted_status

                # get and sanitize hashtags 
                hashtags = status.entities['hashtags']
                hashtag_list = []
                for el in hashtags:
                    hashtag_list.append(el['text'])
                hashtag_count = len(hashtag_list)

                # get and sanitize urls
                urls = status.entities['urls']
                url_list = []
                for el in urls:
                    url_list.append(el['url'])
                url_count = len(url_list)

                # get and sanitize user_mentions
                user_mentions = status.entities['user_mentions']
                mention_list = []
                for el in user_mentions:
                    mention_list.append(el['screen_name'])
                mention_count = len(mention_list)

                # save it all as a tweet
                tweet = [status.id, status.created_at, try_country(status, 'place'), try_city(status, 'place'), status.text.encode('utf-8'), status.lang,
                        hashtag_list, url_list, mention_list, 
                        hashtag_count, url_count, mention_count, 
                        try_get(status, 'possibly_sensitive'),
                        status.favorite_count, status.favorited, status.retweet_count, status.retweeted, 
                        status.user.statuses_count, 
                        status.user.favourites_count, 
                        status.user.followers_count,
                        try_get(status.user, 'description'),
                        try_get(status.user, 'location'),
                        try_get(status.user, 'time_zone')]

                # write to csv
                csvWriter.writerow(tweet)
                
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    # tell us if there's an error
    def on_error(self, status):
        print(status)
        return True
    
     # to handle timeouts
    def on_timeout(self):
        print('timeout occurred')
        return True
    

def read_txt(txt_file):
    ''' reads a .txt file returning a list'''
    food = []
    with open(txt_file) as file:
        while True:
            line = file.readline()
            if not line:
                break
            food.append(line.strip())
    return food


if __name__ == "__main__":
    
    consumer_key = '3XLVvCdmXhpjZYi5pOverZgPL'
    consumer_secret = 'TPVT9NSnQbQbje0yUP1J0wpX0rZHbkBAmx1dhNnBMTSUk4rZZh'
    access_token = '1248331519204249600-ZE16yRQr6xjKt6JLXwgzr2YavcI0ZL'
    access_token_secret = 'JvMHU8aqokwJC0VUOegGTkIhUjY2YQywcsUJOLUOiTuiZ'
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    twitter_stream = MyListener(consumer_key, consumer_secret, access_token, access_token_secret)
    
    # read food list
    food_vocabulary = read_txt('intro2ml_2021_final_project\\food_vocabulary.txt')
    # start scraping twitter
    twitter_stream.filter(track='pizza', languages='en', stall_warnings=True)
    

    '''
    tweetdf=pd.read_csv('intro2ml_2021_final_project\\Data\\all_tweets.csv', names=["id", "created_at", "country", "city", "text", "lang",
                                        "hashtags", "urls", "user_mentions", 
                                        "hashtag_count", "url_count", "mention_count",
                                        "possibly_sensitive", 
                                        "favorite_count", "favorited", "retweet_count", "retweeted",
                                        "user_statuses_count", "user_favorites_count",
                                        "user_follower_count", "user_description", "user_location", "user_timezone"])
    tweetdf.head(10)
    '''