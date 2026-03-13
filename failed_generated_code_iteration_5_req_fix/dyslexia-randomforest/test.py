import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from sklearn.metrics import confusion_matrix

a = np.array([0,0,0,0])
b = np.array([0,0,0,0])

conf=confusion_matrix(a,b,labels=[0,1])
print(conf)

root = tk.Tk()
root.withdraw()
file = filedialog.askopenfilename()
split_tup = os.path.splitext(file)
file_extension = split_tup[1]
dirname= split_tup[0]
basename=os.path.basename(file)
print (dirname)
try:
    # Create target Directory
    os.mkdir(dirname)
    print("Directory ", dirname, " Created ")
except FileExistsError:
    print("Directory ", dirname, " already exists")
newfile=dirname+"/"+basename
from shutil import copyfile
copyfile(file, newfile)

# results=pd.DataFrame(columns=["TN", "FP", "FN", "TP"])
# print (a)
# print(a.shape)
# # a=np.transpose(a)
# # print (a)
# # print(a.shape)
# df = pd.DataFrame(a.reshape(-1, len(a)),columns=["TN", "FP", "FN", "TP"])
# dfb = pd.DataFrame(b.reshape(-1, len(b)),columns=["TN", "FP", "FN", "TP"])
#
# print("hi")
# results=results.append(df,ignore_index=True)
# results=results.append(dfb,ignore_index=True)
# print(results)
# res=results.sum()
# print(res)
# print(type(res), type(results))
# string=res.to_string()
# print(string)
# print(type(string))
# TN=res["TN"]
# print(TN)