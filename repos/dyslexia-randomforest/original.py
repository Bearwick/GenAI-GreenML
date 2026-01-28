# This is a sample Python script.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def confmatrix_of_RandomForest (trainingdata_filepath, testdata_filepath, criterion="entropy"):
    training_data = pd.read_csv(trainingdata_filepath)
    test_data = pd.read_csv(testdata_filepath)

    #drop unnecessary columns
    training_data.drop(['parts {1=akoustiko,2=optiko,3=mousiki}','Subject ID'],axis='columns',inplace=True)
    test_data.drop(['parts {1=akoustiko,2=optiko,3=mousiki}','Subject ID'],axis='columns',inplace=True)

    #replace CN and DYS with 0 and 1
    training_data.replace(to_replace='CN',value=0,inplace=True)
    training_data.replace(to_replace='DYS',value=1,inplace=True)
    test_data.replace(to_replace='CN',value=0,inplace=True)
    test_data.replace(to_replace='DYS',value=1,inplace=True)

    #check for missing values
    training_todrop=training_data.columns[training_data.isna().any()].tolist()
    test_todrop=test_data.columns[test_data.isna().any()].tolist()
    exclude_list= set(training_todrop+test_todrop)
    print(exclude_list)
    training_data.drop(exclude_list,axis='columns',inplace=True)
    test_data.drop(exclude_list,axis='columns',inplace=True)

    ## SERIOUS: convert apo float64 se float32
    training_data=training_data.astype(np.float32,copy=False)
    training_data=training_data.astype({'class': np.int32},copy=False)

    test_data=test_data.astype(np.float32,copy=False)
    test_data=test_data.astype({'class': np.int32},copy=False)


    training_data.to_csv("trainingfinal.csv",index=False)
    test_data.to_csv("testfinal.csv",index=False)

    y_training = training_data['class']
    x_training = training_data.drop(['class'],axis='columns',inplace=False)

    y_test = test_data['class']
    x_test = test_data.drop(['class'],axis='columns',inplace=False)

    #run random forest 5 times (to get the mean of the results)
    a = np.zeros(shape=(5,4))
    for x in range(5):
        model=RandomForestClassifier(criterion=criterion,max_depth=8)
        model.fit(x_training,y_training)
        score = model.score(x_test,y_test)
        print (score)
        y_pred= model.predict(x_test)
        conf = confusion_matrix(y_test, y_pred,labels=[0,1])
        print(conf)
        #convert 2d conf matrix to 1d
        array_1d = conf.flatten()
        a[x]=array_1d
    #get the mean accuracies of the times
    conf = np.mean(np.array([a[0], a[1], a[2], a[3], a[4]]), axis=0)
    print (conf)
    return conf

if __name__ == '__main__':
    conf=confmatrix_of_RandomForest("entire brain_training_2.csv","entire brain_test_2.csv")
    print (conf)