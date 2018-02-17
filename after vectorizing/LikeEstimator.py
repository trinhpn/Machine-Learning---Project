import pickle

import numpy
import pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

CONST_SPLIT_DATA = False
CONST_SPLIT_SIZE = 8000
CONST_BIG5 = ['ope', 'ext', 'con', 'agr', 'neu']
CONST_AGE_MODEL = "nb"
CONST_GENDER_MODEL = "nb"
CONST_PERSONALITY_MODEL = "lr"
CONST_PICKLE_FILENAME = "like_est.pkl"

class Estimator:
    def __init__(self, theDataframe):
        if CONST_AGE_MODEL == "nb":
            self.agePredictor = MultinomialNB()
        elif CONST_AGE_MODEL == "lr":
            self.agePredictor = LinearRegression()

        if CONST_GENDER_MODEL == "nb":
            self.genderPredictor = MultinomialNB()
        elif CONST_GENDER_MODEL == "lr":
            self.genderPredictor = LinearRegression()

        if CONST_PERSONALITY_MODEL == "lr":
            self.personalityPredictor = LinearRegression()

        self.countVectorizer = CountVectorizer()
                
        if CONST_SPLIT_DATA:
            # Calculate number of users (n)
            allIDs = numpy.arange(len(theDataframe))

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
        self.personalityPredictor.fit(X, y)

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
        test_y = self.personalityPredictor.predict(test_X)
        
        resultingDataframe = testDataframe.copy()
        resultingDataframe.loc[:,CONST_BIG5] = test_y

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


