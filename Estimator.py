# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 23:29:49 2018

@author: Trinh
"""
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import metrics
import datetime
import pandas as pd
import numpy as np
CONST_COLS = ['xx-24','25-34','35-49','50-xx']
CONST_BIG5 = ['ope', 'ext', 'con', 'agr', 'neu']
CONST_BASELINE = ['xx-24', 1, 3.91, 3.45, 3.49, 3.58, 2.73]
CONST_SCORES = ['Age Accuracy', 'Gender Accuracy', 'Ope RMSE', 'Neu RMSE', 'Ext RMSE',\
                'Agr RMSE', 'Con RMSE']
CONST_HEADER = ['userid','age',	'gender','ope',	'con','ext','agr','neu']
def main():
    print("Start time: ", datetime.datetime.now())
    # Only run these lines on first run
    profile = pd.read_csv('data/training/profile/profile.csv')
    score_list = []
    #profile = shuffle(profile)
    relation = pd.read_csv('data/training/relation/relation.csv')


    #model = getLikeEstimator()
    #print(predictSingleUser(test_merge_file.loc[10, 'userid'], test_merge_file, model))

    
    for i in range (0,10):
        profile = shuffle(profile)
        train_profile = profile.iloc[:8000,]
        train_merge_file = pd.merge(left=train_profile, right=relation, left_on='userid', right_on='userid')
        del train_merge_file['Unnamed: 0_y']
        del train_merge_file['Unnamed: 0_x']
        
        test_profile = profile.iloc[8001:,]
        model = init_model(train_merge_file)
        test_merge_file = pd.merge(left=test_profile, right=relation, left_on='userid', right_on='userid')
        df = predictDataframe(test_profile, test_merge_file, model)

        
        # Need to change age to bracket of test profiles before calculate accuracy_score
        age_column = test_profile.columns.get_loc("age")
        changeAgeToBracket(test_profile, age_column)
        score = [0,0,0,0,0,0,0]
        score[0] =  accuracy_score(test_profile.age, df.age)
        score[1] =  accuracy_score(test_profile.gender, df.gender)
        score[2] = np.sqrt(metrics.mean_squared_error(test_profile.ope, df.ope))
        score[3] = np.sqrt(metrics.mean_squared_error(test_profile.neu, df.neu))
        score[4] = np.sqrt(metrics.mean_squared_error(test_profile.ext, df.ext))
        score[5] = np.sqrt(metrics.mean_squared_error(test_profile.agr, df.agr))
        score[6] = np.sqrt(metrics.mean_squared_error(test_profile.con, df.con))
        score_list.append(score)
        
    score_report = pd.DataFrame(score_list)
    score_report.to_csv("Score_Report2.csv")
        
    
# Create the 'model' dataframe:
def init_model(merge_file):
    df = merge_file.groupby(['like_id']).mean().reset_index()
    df['count'] = merge_file.groupby(['like_id']).size().reset_index().iloc[:,1]
 
    # Only choose page has more than 2 users like
    df = df.loc[df['count'] > 5]
     #Getting majority gender and age
    for i in range(len(df)):
        # Gender
        if df.iloc[i,2] <= 0.5:
            df.iloc[i,2] = 0
        else:
            df.iloc[i,2] = 1
        # Age    
        df.iloc[i,1] = toBracket(df.iloc[i,1])      
      
    return df

# Return the model dataframe
def getLikeEstimator():
    return pd.read_pickle('model.pkl')  

def predictDataframe(profile_df, merge_test_df, model):

    rows_list = []
    # List through every user in the profile.csv
    for i in range(0, len(profile_df)):
        info = predictSingleUser(profile_df.iloc[i,1], merge_test_df, model)
             
        dict1 = {'userid':profile_df.iloc[i,1], 'age': info[0]\
                 , 'gender': info[1]}
        for k in range (2, len(info)):
            if info[k] != 0:
                dict1[CONST_BIG5[k-2]] = info[k]
            #else: dont need baseline anymore, userpredict handled it
             #   dict1[CONST_BIG5[k-3]] = CONST_BASELINE[k]
        rows_list.append(dict1)
    df = pd.DataFrame(rows_list) 
    return df

# The 3 functions below will take in a df created 
# from function predictDataframe above, return according columns
def predictAge(df):
    return df['age']
def predictGender(df):
    return df['gender']
def predictBig5(df):
    return df[CONST_BIG5]

def predictSingleUser(userid, merge_file, model):
    gender=0.0
    ope = 0.0
    con= 0.0
    ext = 0.0
    agr = 0.0
    neu = 0.0
    user_liked_pages = merge_file[merge_file.userid == userid]['like_id']
    #print(user_liked_pages.count())
    #return user_liked_pages
    if len(user_liked_pages) == 0:
        # Return the baseline prediction
        return 'xx-24', 1, 3.91, 3.45, 3.49, 3.58, 2.73
    #elif model[model.like_id == user_liked_pages.iloc[i]].all(1).any() == False:
        #return 'xx-24', 1, 3.91, 3.45, 3.49, 3.58, 2.73
        
    ages = {}
    
    #return list(model)[0] == 'like_id'
    count = 0
    # Get initial info of all pages
    for page in user_liked_pages:
        #print(page)
        page_info = model[model.like_id == page]
        if not page_info.empty:
            #print(list(page_info))
            gender += page_info.iloc[0,2]
            ope += page_info.iloc[0,3]
            con += page_info.iloc[0,4]
            ext += page_info.iloc[0,5]
            agr += page_info.iloc[0,6]
            neu += page_info.iloc[0,7]
            if page_info.iloc[0,1] in ages:
                ages[page_info.iloc[0,1]] +=1
            else: ages[page_info.iloc[0,1]] =1
            count += 1
    if count == 0:
        return 'xx-24', 1, 3.91, 3.45, 3.49, 3.58, 2.73
    age = max(ages, key=lambda key: ages[key])
    #print(age)
    #page_list.remove([])
    if gender > 0.5: gender = 1
    else: gender = 0
    return [age, gender, round(ope/count,2), round(con/count,2), round(ext/count,2), round(agr/count,2), round(neu/count,2)]
    
        
def changeAgeToBracket(source, column):
    for i in range(0, len(source)):
        source.iloc[i,column] = toBracket(float(source.iloc[i,column]))
     
def toBracket(age):
    if age <= 24:
        return "xx-24"
    elif age >= 25 and age <= 34:
        return "25-34"
    elif age >= 35 and age <= 49:
        return "35-49"
    else: # age >= 50
        return "50-xx"
    
# This function returns a dataframe which consists
# like_id and its age brackets with count            
def getDfAgeCount(source):
    df = source.loc[:,['userid','age', 'like_id']].copy()
    df['age'] = df.age.astype(str)
    changeAgeToBracket(df,1)
    df.groupby(['like_id', 'age']).count().reset_index()
    df.iloc[:,0:3]
    df.rename(columns = {'userid': 'count'}, inplace=True)
    return df

def changeToMajorBracket (page_id, dfagecount):
    df = dfagecount[dfagecount.like_id == page_id]
    
    max = 0
    bracket = ""
    # # List through all the brackets, find the major bracket 
    for i in range(0, len(df)):
        if df.iloc[0,2] > max:
            max = df.iloc[0,2]
            bracket = df.iloc[0,1]
    return bracket
    
if __name__ == "__main__":
    main()    