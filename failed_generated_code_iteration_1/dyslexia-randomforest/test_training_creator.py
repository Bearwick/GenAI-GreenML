import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

#koinws ftiaximo tou training set
def remove_rows_with_this_subjectid (file, subjectid=1):
    split_tup = os.path.splitext(file)
    file_extension = split_tup[1]
    filename = split_tup[0]
    data = pd.read_csv(file)
    index_names = data[data['Subject ID'] == subjectid].index
    data.drop(index_names, inplace=True)
    data.to_csv(filename+"_training_"+str(subjectid)+ ".csv", index=False)

#koinws ftiaximo tou test set
def keep_rows_with_this_subjectid (file,subjectid=1):
    split_tup = os.path.splitext(file)
    file_extension = split_tup[1]
    filename = split_tup[0]
    data = pd.read_csv(file)
    index_names = data[data['Subject ID'] != subjectid].index
    data.drop(index_names, inplace=True)
    data.to_csv(filename+"_test_"+str(subjectid)+ ".csv", index=False)

def choose_file ():
    print("Select csv file")
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename()
    return file

def count_subject_ids(file):
    data = pd.read_csv(file)
    ids = []
    for index,row in data.iterrows():
        subid = int(row['Subject ID'])
        if subid not in ids:
            ids.append(subid)
    print(ids)
    return ids

def create_training_and_test_set (file , subjectid=1):
    remove_rows_with_this_subjectid(file, subjectid)
    keep_rows_with_this_subjectid(file, subjectid)

def change_class_column_numeric_to_nominal (file):
    data = pd.read_csv(file)
    if 'class {0=CN, 1=DYS}' in data.columns:
        data.rename(columns={'class {0=CN, 1=DYS}': 'class'}, inplace=True)
    data['class'] = data['class'].apply(str)
    for index, row in data.iterrows():
        if row['class']=="1":
            data.at[index ,'class']='DYS'
        elif row['class']=="0":
            data.at[index,'class']='CN'
    data.to_csv(file, index=False)

def renamefile(file):
    filename = os.path.basename(file)
    dir = os.path.dirname(file)
    split_tup = os.path.splitext(filename)
    file_extension = split_tup[1]
    name = split_tup[0]
    name = name.replace('_', '-')
    file2 = dir + "/" + name + file_extension
    os.rename(file, file2)
    return file2

def create_all_training_test_sets (file=" "):
    if file==" ":
        file = choose_file()
    file=renamefile(file)
    change_class_column_numeric_to_nominal(file)
    ids = count_subject_ids(file)
    for id in ids:
        create_training_and_test_set(file, id)

if __name__ == '__main__':
    create_all_training_test_sets()
