import pandas as pd
import numpy as np

def process_dataset(dataset = 'agjt'):
    df = pd.read_excel(f'datasets/{dataset}/AJGT.xlsx')

    train_df_init = df.sample(frac=0.9,random_state=200)
    test_df = df.drop(train_df_init.index) 

    train_df = train_df_init.sample(frac=0.9,random_state=200)
    valid_df  = train_df_init.drop(train_df.index)

    def map_to_label(sentiment):
        if sentiment == 'Negative':
            return '0'
        else:
            return '1'

    def write_dataset(df, mode='train'):
        tweets = []
        labels = []

        for id_, record in df.iterrows():
            tweet, sentiment = record["Feed"], record["Sentiment"]
            tweets.append(tweet)
            labels.append(map_to_label(sentiment))
        
        open(f'datasets/ajgt/{mode}_data.txt', 'w').write(('\n').join(tweets))
        open(f'datasets/ajgt/{mode}_labels.txt', 'w').write(('\n').join(labels))

    write_dataset(train_df, 'train')
    write_dataset(valid_df, 'valid')
    write_dataset(test_df,'test')