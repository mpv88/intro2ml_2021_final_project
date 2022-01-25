import tweepy as tw
import pandas as pd
import pickle
import csv



if __name__ == "__main__":
    
    consumer_key = '3XLVvCdmXhpjZYi5pOverZgPL'
    consumer_secret = 'TPVT9NSnQbQbje0yUP1J0wpX0rZHbkBAmx1dhNnBMTSUk4rZZh'
    access_token = '1248331519204249600-ZE16yRQr6xjKt6JLXwgzr2YavcI0ZL'
    access_token_secret = 'JvMHU8aqokwJC0VUOegGTkIhUjY2YQywcsUJOLUOiTuiZ'
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    companies = readNames('intro2ml_2021_final_project\\food_brands.txt')
    downloadTweets(api, companies, file_name='intro2ml_2021_final_project\\Data\\all_tweets_moreUsers_1e5.pkl', min_number_of_followers=1e5)

    df = pd.read_pickle('intro2ml_2021_final_project\\Data\\all_tweets_moreUsers_1e5.pkl')


