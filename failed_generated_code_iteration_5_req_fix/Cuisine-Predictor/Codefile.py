#!/usr/bin/env python
# coding: utf-8

# In[7]:


import json
import os
import pandas as pd
from unidecode import unidecode 


#________________________calculate prior prob of all classes and store in prior_prob[class] = prob    
def prior_probability():
    global classes
    global class_dcount
    classes = list(class_tcount.keys())
    global prior_prob
    for i in classes:
        #docs in class  =  corpus[i]
        # total docs  =  len(data)
        #p(class) = docs in this class / total docs 
        prior_prob[i] = class_dcount[i]/len(data)
        
def vocab_maker():
    #____________________________we will count docs in each class ________________________________________
    global class_dcount
    global data
    for doc in data:
        class_dcount[doc['cuisine']] = class_dcount.get(doc['cuisine'],0)
        class_dcount[doc['cuisine']]  = class_dcount[doc['cuisine']] + 1
    for doc in data:
    # _____________________________________ total tokens in class _____________________________________
        if doc['cuisine'] not in class_tcount:
            #assign class value with tokens of docs
            class_tcount[doc['cuisine']] = len(doc['ingredients'])
        else:
            #update total count of tokens
            class_tcount[doc['cuisine']] += len(doc['ingredients'])
    #_________________________ each ingredient with class and tf ,  here tf = df _______________________
        for i in doc['ingredients']:
            i = unidecode(i.lower())
            if i not in corpus:
                corpus[i] = corpus.get(i,{})
            corpus[i][doc['cuisine']] = corpus[i].get(doc['cuisine'], [])
            corpus[i][doc['cuisine']].append(doc['id'])
    #_________________________ Vocablary Vector (took keys of corpus to form it) ______________________
    vocab = list(corpus.keys())
    #print(len(class_tcount.keys()))  = 20 classes 
    #print(len(vocab))  = 6703 

def file_read():
    global data
    with open("train.json", "r") as handle:
        data = json.load(handle)
    vocab_maker()

    
#______________________________Function to make file of results ___________________________________________
def results_storage():
    #Results are stored in excel sheet
    file_path = "result.xlsx"
    #rearrange to place in 
    data = {"Keys": list(qdoc_class.keys()), "Values": list(qdoc_class.values())}
    # Check if the file exists
    if not os.path.exists(file_path):
        # Create a new Excel file and save the data
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
    else:
        # Load the existing Excel file and add the data
        #df = pd.DataFrame(data)
        #df.to_excel(file_path, index=False)
        print('Data saved in file')

#_________________________________  Naive Baased Approach ___________________________________________
def naive_based(queryset):
    qdoc_class = {}
    for doc in queryset:
        #calculate probability of each doc for all 20 classses
        classes_dprob = {}          
        for c in classes:
            p_ingredient = 1
            #calculate prob of all ingredients in query doc for each class
            for i in doc['ingredients']:
                #for each ingredient calculate its prob
                length = len(corpus.get(i,{}).get(c,[]))
                i_ingredient =(1 + length)/(len(vocab) + class_tcount[c])
                p_ingredient = p_ingredient * i_ingredient
            temp = p_ingredient * prior_prob[c]
            classes_dprob[c] = temp
        #assigning class with max prob to document out of all 20 classes 
        qdoc_class[doc['id']] = max(classes_dprob, key=lambda k:classes_dprob[k])
    return qdoc_class
        #____________________________ Assigning Classes to documents _____________________________________
def testdata():
    global qdoc_class
    with open('test.json','r') as f:
        queryset = json.load(f)
    #Naive based function is used to asses the test data and assign most appropriate class to the document 
        qdoc_class = naive_based(queryset)
    
#_________________________________________check models accuracy_________________________________________
def empirical_accuracy():
    global qdoc_class
    correctly_classified = 0  #variable to store countof correct docs
    with open('train.json','r') as f:
        queryset = json.load(f)
    empirical_dict = naive_based(queryset)
    #now compare the assigned class vs the class doc origionally contains
    l = len(empirical_dict) #total docs 
    for doc in queryset:
        if doc['cuisine'] == empirical_dict[doc['id']]:
            correctly_classified += 1
    empirical_accuracy = correctly_classified / l   # accuracy = true_positive / total documents
    print('Empirical Accuracy is = ',empirical_accuracy,' \nCorrectly classified Document = ',correctly_classified,' and Total Documents = ',l)
    
#_____________________________________________ Main Body __________________________________________

#variables
data = []        #data variable is used to contain all the file data 
corpus = {}        #a comprehensive data {ingredient_name :{ class or origion : docs } } ,,, observation -> each document has ingredient only once so tf = df
class_tcount = {}  #total terms in each class
class_dcount = {}  #documents in each class(origion e.g greek has 100 docs)
vocab = []       #all unique ingredients globally 
prior_prob = {}  #prior probabilities of all the classes(origions e.g greek mexican)
qdoc_class = {}  # test document against assigned class through naive based 
classes = []   #all origions e.g greek , mexican etc

#calling functions 
file_read()
prior_probability()
testdata()
empirical_accuracy()

#for id_doc,class_assigned in qdoc_class.items():
 #   print('Document Id = ',id_doc,' , class -> ',class_assigned)

