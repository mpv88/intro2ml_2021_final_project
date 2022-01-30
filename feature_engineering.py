import pandas as pd

def to_binary(df, col, threshold=1000):
    pass
    
def to_count(df, col):
    pass


if __name__ == "__main__":
    
    tweets_df = pd.read_csv('intro2ml_2021_final_project\\Data\\2k_sample.csv', encoding = 'utf-8')
    
    # transforms variable to binary
    to_binary(tweets_df, variable, threshold=1000)
    
    # transforms variable to count
    to_count(tweets_df, variable)
