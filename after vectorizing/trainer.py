# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:22:09 2018

@author: trinh
"""
import pandas as pd
from LikeEstimator import Estimator


CONST_PATH = '../data/training/'
CONST_BIG5 = ['ope', 'ext', 'con', 'agr', 'neu']

def main():
    profile = pd.read_csv(CONST_PATH + 'profile/profile.csv')
    relation = pd.read_csv(CONST_PATH + 'relation/relation.csv')
    #profile = profile.iloc[:10,:]
    
    train_df = createTrainingDf(profile,relation)
    estimator = Estimator(train_df)
    estimator.pickle()
# Input is the merged file of profile and relation
# Output the dataframe consists 2 columns userid and the string contains all
# the page id
def createUserAndPagesDf(merge_file):
    merge_file['ids'] = merge_file['like_id'] + " "
    df = merge_file.groupby('userid')['ids'].apply(lambda x: x.sum()).reset_index()       
    return df

def createTrainingDf(profile_df, relation_df):
    #profile_df['age'].apply(lambda x: toBracket(x))
    merge_file = pd.merge(left=profile_df, right=relation_df, left_on='userid', right_on='userid')
    del merge_file['Unnamed: 0_y']
    del merge_file['Unnamed: 0_x']
    merge_file['like_id'] = merge_file['like_id'].astype(str)
    pageIDs = createUserAndPagesDf(merge_file)
    training_df = pd.merge(left=profile_df, right=pageIDs, left_on='userid', right_on='userid')
    del training_df['Unnamed: 0']
    training_df['age'] = training_df['age'].apply(toBracket)

    return training_df

def toBracket(age):
    bracket = ""
    if age <= 24.0:
        bracket = "xx-24"
    elif age >= 25.0 and age  <= 34.0:
        bracket = "25-34"
    elif age  >= 35.0 and age  <= 49.0:
        bracket = "35-49"
    else: # age >= 50
        bracket = "50-xx"
    return bracket

if __name__ == "__main__":
    main()  