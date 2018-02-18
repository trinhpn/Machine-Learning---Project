import pandas as pd
import numpy as np
import sys
from LikeEstimator import Estimator
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import metrics
CONST_DEBUG = False
CONST_TEST_PATH = '../data/public-test-data/'
CONST_OUTPUT_PATH = 'output/'
CONST_BIG5 = ['ope', 'ext', 'con', 'agr', 'neu']

def main():
    df = createTestLikeDataframe(CONST_TEST_PATH)
    print(df)
    '''
    profile_df = pd.read_csv(CONST_TEST_PATH + "profile/profile.csv")
    test_profile=profile_df.iloc[8001:,:]    
    estimator = Estimator.unpickle()

    df = estimator.predictAge(df)
    df = estimator.predictGender(df)
    df = estimator.predictPersonality(df)
    
    score = [0,0,0,0,0,0,0]
    score[0] =  accuracy_score(test_profile.age, df.age)
    score[1] =  accuracy_score(test_profile.gender, df.gender)
    score[2] = np.sqrt(metrics.mean_squared_error(test_profile.ope, df.ope))
    score[3] = np.sqrt(metrics.mean_squared_error(test_profile.neu, df.neu))
    score[4] = np.sqrt(metrics.mean_squared_error(test_profile.ext, df.ext))
    score[5] = np.sqrt(metrics.mean_squared_error(test_profile.agr, df.agr))
    score[6] = np.sqrt(metrics.mean_squared_error(test_profile.con, df.con))
    print(score)
    #dataframeToXML(testDF)
    '''
def dataframeToXML(dataframe):
    for index in dataframe.index.tolist()[:]:
        userid = dataframe.get_value(index, 'userid')

        age = dataframe.get_value(index, 'age')

        gender = dataframe.get_value(index, 'gender')
     

        personality = []
        personality.append(dataframe.get_value(index, 'ext'))
        personality.append(dataframe.get_value(index, 'neu'))
        personality.append(dataframe.get_value(index, 'agr'))
        personality.append(dataframe.get_value(index, 'con'))
        personality.append(dataframe.get_value(index, 'ope'))

        dataToXML(userid, age, gender, personality)

def dataToXML(userid, age, gender, personality):
    file = open(CONST_OUTPUT_PATH + userid + ".xml", "w+")
    file.write("<user\n")
    file.write("\tid=\"%s\"\n" % (userid))
    file.write("age_group=\"%s\"\n" % (age))
    file.write("gender=\"%s\"\n" % (gender))
    file.write("extrovert=\"%s\"\n" % (personality[0]))
    file.write("neurotic=\"%s\"\n" % (personality[1]))
    file.write("agreeable=\"%s\"\n" % (personality[2]))
    file.write("conscientious=\"%s\"\n" % (personality[3]))
    file.write("open=\"%s\"\n" % (personality[4]))
    file.write("/>")
    file.close()
    
# Create dataframe containing merged data from
# profile.csv and relation.csv   
def createTestLikeDataframe(path):
    profile_df = pd.read_csv(path + "profile/profile.csv")
    profile_df=profile_df.iloc[8001:,:]
    relation_df = pd.read_csv(path + "relation/relation.csv")
    merge_file = pd.merge(left=profile_df, right=relation_df, left_on='userid', right_on='userid')
    cols = ['age', 'gender','Unnamed: 0_y','Unnamed: 0_x'] + CONST_BIG5 
    merge_file.drop(cols, inplace=True, axis=1)

    merge_file['like_id'] = merge_file['like_id'].astype(str)
    pageIDs = createUserAndPagesDf(merge_file)
    training_df = pd.merge(left=profile_df, right=pageIDs, left_on='userid', right_on='userid')
    del training_df['Unnamed: 0']    
    print(training_df)
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
