import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def split_to_part_files (file=" "):
    if file==" ":
        print("Select csv file ")
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename()
    split_tup = os.path.splitext(file)
    file_extension = split_tup[1]
    filename = split_tup[0]
    data = pd.read_csv(file)
    index_names1 = data[data['parts {1=akoustiko,2=optiko,3=mousiki}'] != 1].index
    data1=data.drop(index_names1)
    index_names2 = data[data['parts {1=akoustiko,2=optiko,3=mousiki}'] != 2].index
    data2 = data.drop(index_names2)
    index_names3 = data[data['parts {1=akoustiko,2=optiko,3=mousiki}'] != 3].index
    data3 = data.drop(index_names3)
    data1.to_csv(filename+"partakoustiko" + ".csv", index=False)
    data2.to_csv(filename + "partoptiko" + ".csv", index=False)
    data3.to_csv(filename + "partmousiki" + ".csv", index=False)

def split_files_to_part_files():
    print("choose files to be splitted")
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames()
    for file in files:
        split_tup = os.path.splitext(file)
        filename= split_tup[0]
        file_extension = split_tup[1]
        if (file_extension == ".xlsx"):
            read_file = pd.read_excel(file)
            print(read_file)
            file=filename+".csv"
            read_file.to_csv(file, index=None, header=True)
        split_to_part_files(file)


if __name__ == '__main__':
    split_files_to_part_files()