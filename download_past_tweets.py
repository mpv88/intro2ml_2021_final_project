import tweepy as tw
import pandas as pd
import pickle

def downloadTweets(api, users, file_name = 'all_tweets.pkl', min_number_of_followers=1e6):
    ''' downloads number 'count' tweets from users listed in 'users' 

        Parameters
        ----------
        api: twitter API object for download of tweets
        users: list of target twitter accounts
        file_name: name of file on which data are saved
        min_number_of_followers: lower bound to the number of followers of the given users from which
                                 one wants to download tweets

        Returns
        -------
        
    '''
    
    all_tweets = []
    tot_tweets_dwlded = 0

    # download tweets of given users
    print("downloading...")
    for user in users:

        user_object = api.get_user(screen_name = user)

        if user_object.followers_count >= min_number_of_followers:
            tweets = tw.Cursor(api.user_timeline, screen_name = user, tweet_mode="extended").items(3200)
            #tweets = api.user_timeline(screen_name = user, count=20)
            new_tweets = []
            for status in tweets:
                new_tweets += [status]
            
            all_tweets.extend(new_tweets)
            tweets_dwlded = len(new_tweets)
            tot_tweets_dwlded += tweets_dwlded
            print('\t user {} -- {} tweets downloaded'.format(user, tweets_dwlded))
            print('\t\t we have {} tweets now...'.format(tot_tweets_dwlded))

    # saving to JSON
    print("saving tweets to file...")
    with open(file_name,'wb') as file: 
        pickle.dump(all_tweets, file)


def readNames(file_txt):
    ''' substitutes the date/time of creation of a tweet with a categorical variable indicate a time zone duringe the day
        Parameters
        ----------
        file_txt: file containing names of interest

        Returns
        -------
        names: list of the names contained in the given file
    '''

    # companies list
    names = []
    with open(file_txt) as file:
        while True:
            line = file.readline()
            if not line:
                break
            names.append(line.strip())
    
    return names



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


