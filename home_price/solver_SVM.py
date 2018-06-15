"""Kaggle competition: titanic
Titanic: Machine Learning from Disaster
Start here! Predict survival on the Titanic and get familiar with ML basics
"""
import os
import argparse
import csv
import numpy as np
import pandas as pd

from sklearn import svm

def clean_data(df):
    return df


def main(args):
    train_file = os.path.join(args.data_dir, 'train.csv')
    train_df = pd.read_csv(train_file)
    train_df = clean_data(train_df)
    print(train_df.info())

    feature_columns = ['MSSubClass', 'LotArea', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt']

    features = train_df[feature_columns].values
    targets = train_df['SalePrice'].values

    clf = svm.NuSVR()
    clf.fit(features, targets)

    test_file = os.path.join(args.data_dir, 'test.csv')
    test_df = pd.read_csv(test_file)
    test_df = clean_data(test_df)
    print(test_df.info())

    features = test_df[feature_columns].values
    predicts = clf.predict(features)
    ids = test_df['Id'].values

    with open('/tmp/kaggle_submit.csv', 'w') as fileobj:
        writer = csv.writer(fileobj)
        writer.writerow(['Id', 'SalePrice'])
        for id, price in zip(ids, predicts):
            writer.writerow([id, price])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='The directory where the input data is stored.')

    args = parser.parse_args()

    main(args)
