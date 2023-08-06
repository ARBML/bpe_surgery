import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
    
def process_dataset(dataset = 'ajgt'):
    if dataset == 'ajgt':
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

    elif dataset == 'labr':
        base_path = f'datasets/{dataset}'
        def map_label(lbl):
            if int(lbl) >= 4:
                return '1'
            else:
                return '0'

        def write_dataset(path, mode='train'):
            labels = []
            reviews_text = []
            reviews = []

            with open(f'{base_path}/reviews.tsv', encoding="utf-8") as tsvfile:
                tsvreader = csv.reader(tsvfile, delimiter="\t")
                for line in tsvreader:
                    reviews.append(line)

            with open(f'{base_path}/{path}', encoding="utf-8") as f:
                for id_, record in enumerate(f.read().splitlines()):
                    rating, _, _, _, review_text = reviews[int(record)]
                    if rating == '3':
                        continue
                    labels.append(map_label(rating))
                    reviews_text.append(review_text)     

            if mode == 'train':
                X_train, X_test, y_train, y_test = train_test_split(reviews_text, labels, test_size=0.05, random_state=200)
                open(f'{base_path}/{mode}_data.txt', 'w').write(('\n').join(X_train))
                open(f'{base_path}/{mode}_labels.txt', 'w').write(('\n').join(y_train))

                open(f'{base_path}/valid_data.txt', 'w').write(('\n').join(X_test))
                open(f'{base_path}/valid_labels.txt', 'w').write(('\n').join(y_test))

            else:  
                open(f'{base_path}/{mode}_data.txt', 'w').write(('\n').join(reviews_text))
                open(f'{base_path}/{mode}_labels.txt', 'w').write(('\n').join(labels))
        write_dataset('2class-balanced-train.txt', 'train')
        write_dataset('2class-balanced-test.txt','test')