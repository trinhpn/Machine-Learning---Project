import pandas as pd
import sys

CONST_DEBUG = False
CONST_TEST_PATH = sys.argv[1]
CONST_OUTPUT_PATH = sys.argv[2]
CONST_BIG5 = ['ope', 'ext', 'con', 'agr', 'neu']
CONST_COLS = ['xx-24','25-34','35-49','50-xx']

def main():
    profile = pd.read_csv(CONST_TEST_PATH + 'profile/profile.csv')
    
    merge_file = createLikeDataframe(CONST_TEST_PATH)
    estimator = pd.read_pickle("like_model_feb13.pkl")
    df=predictDataframe(profile, merge_file, estimator)
    print(df)
    dataframeToXML(df)

def predictDataframe(profile_df, merge_test_df, model):

    rows_list = []
    # List through every user in the profile.csv
    for i in range(0, len(profile_df)):
        info = list(predictSingleUser(profile_df.iloc[i,1], merge_test_df, model))
        
        # Assign baseline if 0 result
        age = ''
        gender = 0
        if info[0] == '0':
            age = 'xx-24'
            gender = 1
        else:
            age = info[0]
            gender = info[1]
            
        dict1 = {'userid':profile_df.iloc[i,1], 'age': age\
                 , 'gender': gender}
        for k in range (2, len(info)):
            if info[k] != 0:
                dict1[CONST_BIG5[k-2]] = info[k]
            else:
                dict1[CONST_BIG5[k-2]] = CONST_BASELINE[k]

        rows_list.append(dict1)
    df = pd.DataFrame(rows_list) 
    return df

def predictSingleUser(userid, merge_file, model):
    gender=0.0
    ope = 0.0
    con= 0.0
    ext = 0.0
    agr = 0.0
    neu = 0.0
    user_liked_pages = merge_file[merge_file.userid == userid]['like_id']
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

def dataframeToXML(dataframe):
    for index in dataframe.index.tolist()[:]:
        userid = dataframe.get_value(index, 'userid')

        age = dataframe.get_value(index, 'age')
        gender = dataframe.get_value(index, 'gender')

        '''
        if age < 25.0:
            ageBracket = "xx-24"
        elif age >= 25.0 and age < 35.0:
            ageBracket = "25-34"
        elif age >= 35.0 and age < 50.0:
            ageBracket = "35-49"
        else: # age >= 50.0:
            ageBracket = "50-xx"

        gender = dataframe.get_value(index, 'gender')
        if gender >= 0.5:
            genderString = "female"
        else: # gender < 0.5
            genderString = "male"
        '''    
        personality = []
        personality.append(dataframe.get_value(index, 'ext'))
        personality.append(dataframe.get_value(index, 'neu'))
        personality.append(dataframe.get_value(index, 'agr'))
        personality.append(dataframe.get_value(index, 'con'))
        personality.append(dataframe.get_value(index, 'ope'))

        dataToXML(userid, age, gender, personality)

def dataToXML(userid, age, gender, personality):
    file = open(sys.argv[2] + userid + ".xml", "w+")
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
def createLikeDataframe(path):
    profile = pd.read_csv(path + "profile/profile.csv")
    relation = pd.read_csv(path + "relation/relation.csv")
    merge_file = pd.merge(left=profile, right=relation, left_on='userid', right_on='userid')
    del merge_file['Unnamed: 0_y']
    del merge_file['Unnamed: 0_x']
    return merge_file



if __name__ == "__main__":
    main()
