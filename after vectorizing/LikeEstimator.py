"""
@author: Mai Pham
"""
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cross_validation import cross_val_score
CONST_SPLIT_DATA = False
CONST_SPLIT_SIZE = 8000
CONST_BIG5 = ['ope', 'ext', 'con', 'agr', 'neu']
CONST_AGE_MODEL = "nb"
CONST_GENDER_MODEL = "nb"
CONST_PERSONALITY_MODEL = "svr"
CONST_PICKLE_FILENAME = "like_est.pkl"

class Estimator:
    def __init__(self, theDataframe):
        clf1 = MultinomialNB()
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = BernoulliNB()
        clf4 = KNeighborsClassifier(n_neighbors=5)
        if CONST_AGE_MODEL == "nb":
            self.agePredictor = MultinomialNB()
        elif CONST_AGE_MODEL == "lr":
            self.agePredictor = LinearRegression()
        elif CONST_AGE_MODEL == "voting":
            self.agePredictor = VotingClassifier(estimators=[('mnb', clf1), ('rf', clf2), ('knn', clf4)], voting='hard')
            
        if CONST_GENDER_MODEL == "nb":
            self.genderPredictor = MultinomialNB()
        elif CONST_GENDER_MODEL == "lr":
            self.genderPredictor = LinearRegression()
        elif CONST_GENDER_MODEL == "voting":
            self.genderPredictor = VotingClassifier(estimators=[('mnb', clf1), ('rf', clf2), ('bnb', clf3)], voting='hard')
            
        if CONST_PERSONALITY_MODEL == "lr":
            self.personalityPredictor = LinearRegression()
        elif CONST_PERSONALITY_MODEL == "svr":
            self.svrOpePredictor = SVR(kernel='rbf')
            self.svrExtPredictor = SVR(kernel='rbf')
            self.svrConPredictor = SVR(kernel='rbf')
            self.svrAgrPredictor = SVR(kernel='rbf')
            self.svrNeuPredictor = SVR(kernel='rbf')
        self.countVectorizer = CountVectorizer()
                
        if CONST_SPLIT_DATA:
            # Calculate number of users (n)
            allIDs = np.arange(len(theDataframe))

            # Split data into the users from 0 to SPLIT_DATA_SIZE
            # for training and the rest of the users for testing.
            trainIDs = allIDs[:CONST_SPLIT_SIZE]
            trainData = theDataframe.loc[trainIDs, :]

            # Create train_X and train_y's based on training data.
            train_X = self.countVectorizer.fit_transform(trainData['ids'])
            ageTrain_y = trainData['age']
            genderTrain_y = trainData['gender']
            personalityTrain_y = trainData.loc[:, CONST_BIG5]
        else:
            # Create train_X and train_y's based on entire dataset.
            train_X = self.countVectorizer.fit_transform(theDataframe['ids'])
            ageTrain_y = theDataframe['age']
            genderTrain_y = theDataframe['gender']
            personalityTrain_y = theDataframe.loc[:, CONST_BIG5]

        self.trainAge(train_X, ageTrain_y)
        self.trainGender(train_X, genderTrain_y)
        self.trainPersonality(train_X, personalityTrain_y)

    def trainAge(self, X, y):
        self.agePredictor.fit(X, y)

    def trainGender(self, X, y):
        self.genderPredictor.fit(X, y)

    def trainPersonality(self, X, y):
        if CONST_PERSONALITY_MODEL != "svr":
            self.personalityPredictor.fit(X, y)
        else:
            self.svrOpePredictor.fit(X, y.loc[:,'ope'])
            self.svrExtPredictor.fit(X, y.loc[:,'ext'])
            self.svrConPredictor.fit(X, y.loc[:,'con'])
            self.svrAgrPredictor.fit(X, y.loc[:,'agr'])
            self.svrNeuPredictor.fit(X, y.loc[:,'neu'])
            
    def predictAge(self, testDataframe):
        test_X = self.countVectorizer.transform(testDataframe['ids'])
        test_y = self.agePredictor.predict(test_X)

        resultingDataframe = testDataframe.copy()
        resultingDataframe['age'] = test_y

        return resultingDataframe

    def predictGender(self, testDataframe):
        test_X = self.countVectorizer.transform(testDataframe['ids'])
        test_y = self.genderPredictor.predict(test_X)

        resultingDataframe = testDataframe.copy()
        resultingDataframe['gender'] = test_y

        return resultingDataframe

    def predictPersonality(self, testDataframe):
        test_X = self.countVectorizer.transform(testDataframe['ids'])
        resultingDataframe = testDataframe.copy()

        if CONST_PERSONALITY_MODEL != "svr":
            test_y = self.personalityPredictor.predict(test_X)
            resultingDataframe.loc[:,CONST_BIG5] = test_y
        else:
            test_ope = self.svrOpePredictor.predict(test_X)
            test_ext = self.svrExtPredictor.predict(test_X)
            test_con = self.svrConPredictor.predict(test_X)
            test_agr = self.svrAgrPredictor.predict(test_X)
            test_neu = self.svrNeuPredictor.predict(test_X)
            resultingDataframe.loc[:,'ope'] = test_ope
            resultingDataframe.loc[:,'ext'] = test_ext
            resultingDataframe.loc[:,'con'] = test_con
            resultingDataframe.loc[:,'agr'] = test_agr
            resultingDataframe.loc[:,'neu'] = test_neu

        return resultingDataframe

    # Pickles the current Estimator object.
    # Save a "txt_est.pkl" file to the current directory.
    # CAUTION: This overwrites any existing file in the current directory
    #          named txt_est.pkl.
    def pickle(self):
        pickleFile = open(CONST_PICKLE_FILENAME, 'wb')
        pickle.dump(self, pickleFile)

    # Unpickles the "txt_est.pkl" file in the current directory.
    # Returns the Estimator object that was pickled.
    def unpickle():
        pickleFile = open(CONST_PICKLE_FILENAME, 'rb')
        return pickle.load(pickleFile)

'''
    def validate_model(df, ageModel):
        kf = KFold(n_splits=3, shuffle=True, random_state=1)
        scores = list()
        cv = CountVectorizer()
        
        for train, test in kf.split(df):
            train_data = df.loc[train,:]
            test_data = df.loc[test,:]
            #print(train_data)
            #print(test_data)
            train_X = cv.fit_transform(train_data['ids'])
            ageTrain_y = train_data['age']
            test_X = cv.fit_transform(test_data['ids'])
            ageTest_y = test_data['age']
            model = ageModel.fit(train_X, ageTrain_y)
            print(model.predict(test_X))
        #print(scores)
'''