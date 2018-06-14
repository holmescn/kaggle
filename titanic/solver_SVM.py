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

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SUBMISSION_FILE = 'submit.csv'


def read_df(args, filename):
    df = pd.read_csv(os.path.join(args.data_dir, filename))

    sex2num = lambda s: 1 if s == 'male' else 2

    df['Sex'] = [sex2num(sex) for sex in df['Sex'].values]
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    return df


def main(args):
    train_df = read_df(args, TRAIN_FILE)

    features = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
    labels = train_df['Survived'].values

    clf = svm.NuSVC()
    clf.fit(features, labels)

    test_df = read_df(args, TEST_FILE)

    features = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
    passanger_ids = test_df['PassengerId'].values

    with open(os.path.join(args.data_dir, SUBMISSION_FILE), 'w') as fileobj:
        writer = csv.writer(fileobj)
        writer.writerow(['PassengerId', 'Survived'])
        for passanger_id, survived in zip(passanger_ids, clf.predict(features)):
            writer.writerow([passanger_id, survived])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='The directory where the input data is stored.')

    args = parser.parse_args()

    main(args)
