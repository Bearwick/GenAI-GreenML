from os import listdir
import os
from os.path import isfile, join
import tkinter as tk
from tkinter import filedialog
import RandomForestClassification as rd
import pandas as pd
import test_training_creator as dys
import splitparts

# creates training-test sets for leave one out cross validation,runs random forest for each set,
# writes csv with confusion matrix results and ac,sens,spec and returns dictionary of [filename,ac,sens,spec]
def organizer(file_path=" "):
    #if file not passed as parameter, open file dialog to get file
    if file_path == " ":
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
    print("hi")
    dys.create_all_training_test_sets(file=file_path)
    folder_path = os.path.dirname(file_path)
    print(folder_path)

    # make a list of all filenames in folder (which end in .csv)
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.lower().endswith(('.csv'))]
    print(onlyfiles)

    results = pd.DataFrame(columns=["TN", "FP", "FN", "TP"])

    #iterate through all training-test sets and perform randomforest
    for file in onlyfiles:
        split = file.split("_", 2)
        if len(split) == 3:
            typeof = split[1]
            # print(typeof)
            if typeof == "training":
                test_file = split[0] + "_test_" + split[2]
                print(test_file)
                if test_file in onlyfiles:
                    wholetraining = folder_path + '/' + file
                    wholetest = folder_path + '/' + test_file
                    print(wholetest, wholetraining)
                    conf = rd.confmatrix_of_RandomForest(wholetraining, wholetest)
                    print(conf)
                    dconf = pd.DataFrame(conf.reshape(-1, len(conf)), columns=["TN", "FP", "FN", "TP"])
                    results = results.append(dconf, ignore_index=True)

    # sum the total True positives, true negatives etc and calculate Accuracy, sensitivity specificity
    totals = results.sum()
    TP = totals["TP"]
    TN = totals["TN"]
    FP = totals["FP"]
    FN = totals["FN"]
    print(TP, TN, FP, FN, type(TP))
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    totals = totals.to_string()

    #write confusion matrixes to csv
    towrite = folder_path + "/confmatrixresults" + split[0] + ".csv"
    results.to_csv(towrite, index=False)

    #apend at the end of the csv the Accuracy, sensitivity, specificity
    with open(towrite, "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("\n")
        file_object.write("TP: " + "{:.2f}".format(TP) + " TN: " + "{:.2f}".format(TN) + " FP: " + "{:.2f}".format(
            FP) + " FN: ""{:.2f}".format(FN) + "\n")
        file_object.write("\n")
        file_object.write("accuracy: " + "{:.2f}".format(accuracy) + "\n")
        file_object.write("sensitivity: " + "{:.2f}".format(sensitivity) + "\n")
        file_object.write("specificity: " + "{:.2f}".format(specificity) + "\n")

    #what this function returns: ["name"]= name of file processed, ["AC"]= accuracy,["SEN"]= sensitivity,["SP"]= specificity
    to_return = dict()
    name=os.path.basename(file_path)
    name=os.path.splitext(name)[0]
    to_return["name"]=name
    to_return["AC"] = accuracy
    to_return["SEN"] = sensitivity
    to_return["SP"] = specificity
    return to_return

# takes a file A and creates a folder name A with the file A in it
# Why?: To keep each experiment in different folder for ease of read
def from_file_create_folder_with_file(file=" "):
    if file==" ":
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename()
    split_tup = os.path.splitext(file)
    file_extension = split_tup[1]
    dirname = split_tup[0]
    basename = os.path.basename(file)
    print(dirname)
    try:
        # Create target Directory
        os.mkdir(dirname)
        print("Directory ", dirname, " Created ")
    except FileExistsError:
        print("Directory ", dirname, " already exists")
    newfile = dirname + "/" + basename
    from shutil import copyfile
    copyfile(file, newfile)
    return newfile

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title="Choose all datasets to do Random-Forest")
    results= pd.DataFrame(columns = ["name","AC","SEN","SP"])
    for file in files:
        newfile=from_file_create_folder_with_file(file)
        result=organizer(newfile)
        results=results.append(result,ignore_index=True)
    dir=os.path.dirname(files[0])
    filename=dir+"/"+"totalresults.csv"
    results.to_csv(filename,index=False)
    # organizer()
