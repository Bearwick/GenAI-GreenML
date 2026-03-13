# Programmed by Aspen Henry
# This program uses two .csv files within its directory (trainging data and test sample data)
# to classify test samples using the K Nearest Neighbor algorithm.
import math

#This function is used to preformat the csv files for use
#It assumes that the csv is in the form [class_label, a1, a2, ..., an]
#with the first row of data in the csv being lables for columns
def preformat(fileName):
    with open(fileName) as file:
        contents = file.readlines()
        for i in range(len(contents)):
            contents[i] = contents[i][:-1]
            contents[i] = contents[i].split(',')

        for i in range(1, len(contents)):
            for j in range(len(contents[i])):
                contents[i][j] = int(contents[i][j])

    return contents

#Function for calculating the Euclidean Distance
def getDistance(x1, x2):
    distance = 0
    for i in range(1, len(x1)):
        distance += math.pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


#Function for getting the output class of the test sample with KNN
def KNN(trainingData, tup, k):
    neighborDistances = [20000]*k
    neighborClasses = [None]*k

    #Calculating the k closest distances and storing the corresponding classes
    for data in trainingData:
        if(isinstance(data[0], str)):
            continue

        distance = getDistance(tup, data)
        if(all(i < distance for i in neighborDistances)):
            continue
        else:
            del neighborClasses[neighborDistances.index(max(neighborDistances))]
            neighborClasses.append(data[0])
            neighborDistances.remove(max(neighborDistances))
            neighborDistances.append(distance)

    #Calculating the votes (weights) for each class by using a summation of (1 / distance)
    classVotes = {}
    for i in range(len(neighborClasses)):
        if (neighborClasses[i] not in classVotes.keys()):
            classVotes[neighborClasses[i]] = (1 / neighborDistances[i])
        else:
            classVotes[neighborClasses[i]] += (1 / neighborDistances[i])

    for cj, weight in classVotes.items():
        if (weight == max(classVotes.values())):
            return cj

#Driver function for performing the analysis and classification
def main():
    trainingFileName = "MNIST_train.csv"
    trainingData = preformat(trainingFileName)

    testFileName = "MNIST_test.csv"
    testData = preformat(testFileName)

    k = 7

    #Classifying test data and finding statistics for analysis
    desiredClasses = []
    computedClasses = []

    for test in testData:
        if(isinstance(test[0], str)):
            continue

        desiredClasses.append(test[0])
        computedClasses.append(KNN(trainingData, test, k))

    correctClassifications = 0;
    totalClassifications = 0;
    for i in range(len(desiredClasses)):
        totalClassifications += 1
        if (desiredClasses[i] == computedClasses[i]):
            correctClassifications += 1

    accuracy = (correctClassifications / totalClassifications) * 100
    missedClassifications = totalClassifications - correctClassifications

    #Printing the output
    print("\nK = " + str(k) + '\n')
    for i in range(len(desiredClasses)):
        print("Desired class: " + str(desiredClasses[i]) +
              " computed class: " + str(computedClasses[i]))
    print("\nAccuracy rate: " + str(accuracy) + "%" +'\n')
    print("Number of misclassified test samples: " + str(missedClassifications) + '\n')
    print("total number of test samples: " + str(totalClassifications))
    #print(KNN(trainingData, testData[34], k))


if __name__ == "__main__":
    main()
