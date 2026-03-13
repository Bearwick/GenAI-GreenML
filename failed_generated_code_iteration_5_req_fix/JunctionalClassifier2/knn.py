from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import numpy as np
import csv
import copy
import unittest


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4)
features = []  # This variable is used to hold features during training
labels = []  # This variable is used to hold labels during training
pFeatures = []  # This variable is used to hold unsupervised features during prediction/queries


def makeCSV(inpt):
    """ This method is used to extract data from a CSV with the supervised class labels, this is for training purposes
    'inpt' is the filename of the CSV to extract from """
    try:
        first = True
        with open(inpt, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                cols = row[0].split(",")
                if(cols[len(cols)-1] == ''):
                    cols.pop()
                if(first == False):
                    both = np.asarray(cols, dtype=np.double).tolist()
                    # print(both)
                    num = both.pop()
                    if(num > 0):
                        num = 1
                    elif(num < 0):
                        num = -1
                        # print(num)
                    labels.append(num)
                    features.append(both)
                else:
                    first = False
                    # we want to ignore the initial headings line, so do nothing
    except:
        print("There was an error reading from the csv, did you:\n - Provide the correct name for the CSV?\n - Have the correct CSV format?\n - Put the CSV in the same directory as this file?")


def trainAndTestModel():
    """ This method trains and tests the model by partitioning the input data into training and testing sets and then outputs the accuracy """
    makeCSV('14k.csv')
    feat_train, feat_test, lab_train, lab_test = train_test_split(
        features, labels, test_size=0.3)
    classifier.fit(feat_train, lab_train)  # This line trains the classifier
    # This line runs a prediction on the testing set

    y_pred = classifier.predict(feat_test)
    print("Accuracy:", metrics.accuracy_score(lab_test, y_pred))


def loadModel():
    """ This method loads the trained model """
    global classifier
    pickle_in = open("dict.pickle", "rb")
    classifier = pickle.load(pickle_in)


def saveModel():
    """ This method persists the trained model """
    global classifier
    pickle_out = open("dict.pickle", "wb")
    pickle.dump(classifier, pickle_out)
    pickle_out.close()


def wipeVariables():
    """ This is a utility method to empty the feature and label lists """
    global features
    global labels
    features = []
    labels = []


def predict():
    """ This method will take the input data from the input csv file and query the classifier on the data """
    loadModel()
    takeInput()
    prediction = classifier.predict(pFeatures)
    print(prediction)
    saveModel()


def takeInput():
    """ This method extracts input features from the input csv file """
    try:
        first = True
        with open('input.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                cols = row[0].split(",")
                if(cols[len(cols)-1] == ''):
                    cols.pop()
                if(first == False):
                    ftrs = np.asarray(cols, dtype=np.double).tolist()
                    pFeatures.append(ftrs)
                else:
                    first = False
                    # we want to ignore the initial headings line, so do nothing
    except:
        print("There was an error reading from the csv, did you:\n - Provide the correct name for the CSV?\n - Have the correct CSV format?\n - Put the CSV in the same directory as this file?")


# loadModel()
# trainAndTestModel()
# saveModel()
predict()


class Testing(unittest.TestCase):
    # Run 'python -m unittest knn.Testing' from the command line to run the tests
    def setUp(self):
        wipeVariables()
        return super().setUp()

    def tearDown(self):
        wipeVariables()
        return super().tearDown()

    def test_MakeCSV4(self):
        """ This test makes sure that features are correctly extracted and isolated from labels """
        makeCSV('test.csv')
        try:
            self.assertEqual(len(features), 4)
            print("Test 1 passed")
        except:
            print("Test 1 failed")

    def test_MakeCSV2(self):
        """ This test makes sure that labels are correctly extracted and isolated from features """
        makeCSV('test2.csv')
        try:
            self.assertEqual(len(labels), 2)
            print("Test 2 passed")
        except:
            print("Test 2 failed")
