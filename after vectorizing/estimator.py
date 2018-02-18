"""
@author: Mai Pham
"""
import numpy as np
import sys
import pandas as pd
from LikeEstimator import Estimator


CONST_DEBUG = False
CONST_TEST_PATH = sys.argv[1]
CONST_OUTPUT_PATH = sys.argv[2]
CONST_BIG5 = ['ope', 'ext', 'con', 'agr', 'neu']

def main():
    profile_df = pd.read_csv(CONST_TEST_PATH + "profile/profile.csv")
    relation_df = pd.read_csv(CONST_TEST_PATH + "relation/relation.csv")
    testDF = createTestLikeDataframe(profile_df, relation_df)

    estimator = Estimator.unpickle()

    testDF = estimator.predictAge(testDF)
    testDF = estimator.predictGender(testDF)
    testDF = estimator.predictPersonality(testDF)
    
    dataframeToXML(testDF)

def dataframeToXML(dataframe):
    for index in dataframe.index.tolist()[:]:
        userid = dataframe.get_value(index, 'userid')

        age = dataframe.get_value(index, 'age')
        gender = dataframe.get_value(index, 'gender')
        personality = []
        personality.append(round(dataframe.get_value(index, 'ext'),2))
        personality.append(round(dataframe.get_value(index, 'neu'),2))
        personality.append(round(dataframe.get_value(index, 'agr'),2))
        personality.append(round(dataframe.get_value(index, 'con'),2))
        personality.append(round(dataframe.get_value(index, 'ope'),2))

        dataToXML(userid, age, gender, personality)

def dataToXML(userid, age, gender, personality):
    file = open(CONST_OUTPUT_PATH + userid + ".xml", "w+")
    file.write("<user\n")
    file.write("\tid=\"%s\"\n" % (userid))
    file.write("age_group=\"%s\"\n" % (age))
    file.write("gender=\"%s\"\n" % (gender))
    file.write("extrovert=\"%s\"\n" % (round(personality[0],2)))
    file.write("neurotic=\"%s\"\n" % (round(personality[1],2)))
    file.write("agreeable=\"%s\"\n" % (round(personality[2],2)))
    file.write("conscientious=\"%s\"\n" % (round(personality[3],2)))
    file.write("open=\"%s\"\n" % (round(personality[4],2)))
    file.write("/>")
    file.close()
    
# Create dataframe containing merged data from
# profile.csv and relation.csv   
def createTestLikeDataframe(profile_df, relation_df):
    
    merge_file = pd.merge(left=profile_df, right=relation_df, left_on='userid', right_on='userid')
    cols = ['age', 'gender','Unnamed: 0_y','Unnamed: 0_x'] + CONST_BIG5 
    merge_file.drop(cols, inplace=True, axis=1)

    merge_file['like_id'] = merge_file['like_id'].astype(str)
    pageIDs = createUserAndPagesDf(merge_file)
    training_df = pd.merge(left=profile_df, right=pageIDs, left_on='userid', right_on='userid')
    del training_df['Unnamed: 0']    
    return training_df

def createUserAndPagesDf(merge_file):
    merge_file['ids'] = merge_file['like_id'] + " "
    df = merge_file.groupby('userid')['ids'].apply(lambda x: x.sum()).reset_index()       
    return df

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
