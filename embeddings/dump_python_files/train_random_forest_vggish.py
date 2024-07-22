import os
import glob
import numpy as np
import pandas as pd

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import pickle 
import multiprocessing
from datetime import datetime
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold 
#input is a dict with embeddings for every minute 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

def merge_dicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def merge_list_of_dicts(list):
    null_dict= {}
    for dict in list:
        null_dict= merge_dicts(null_dict, dict)
    return null_dict

storage= "d:\\Research\\analyze_embeddings\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
wd= "D:\\github\\man-vs-BirdNet-summer-2024-\\embeddings"
common_resources= wd+ "\\common_resources\\"
dir= common_resources+ "store_vggish_embeddings\\"
conf=0.25

os.chdir(dir)
list_of_dicts= glob.glob("*.pickle")
# print(list_of_dicts)
array_of_dicts=[]
for dicts in list_of_dicts:
    with open(dicts, 'rb') as f:
        dictionary = pickle.load(f)
    array_of_dicts.append(dictionary)

merged_dictionary= merge_list_of_dicts(array_of_dicts)

os.chdir(common_resources)
with open("y_arr.pickle", 'rb') as f:
    y_dictionary = pickle.load(f)
# print(y_dictionary)
# print(merged_dictionary)

subset_merged_dict= {i:merged_dictionary[i] for i in list(y_dictionary.keys()) if i in merged_dictionary}
subset_y_dict= {i: y_dictionary[i] for i in list(subset_merged_dict.keys()) if i in y_dictionary}
x= list(subset_merged_dict.values())
x= [list(i) for i in x]
y= list(subset_y_dict.values())
y= [list(i) for i in y]
print(len(y), len(x))

def compare_array(array1, array2):
    for i in range(len(array1)):
        if array1[i]==array2[i]:
            continue
        else:
            return False
    return True

def score_list_of_arrays(list1, list2):
    counter_correct= 0
    denom= 0
    for i in range(len(list1)):
        denom+=1
        if compare_array(list1[i],list2[i]):
            counter_correct+=1
        else:
            continue
    return counter_correct/denom

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

t1= datetime.now()
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_jobs=4)

from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
kf.get_n_splits(x)

print(kf)  
list= []
counter=0
for train_index, test_index in kf.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
    y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train) 
    # print("ACCURACY OF THE MODEL (test):", metrics.accuracy_score(y_test, y_pred)) 
    # print("ACCURACY OF THE MODEL (train):", metrics.accuracy_score(y_train, y_train_pred)) 
    score= score_list_of_arrays(y_test, y_pred)
    print("ACCURACY OF THE MODEL* (test): Fold number "+str(counter)+" :", score_list_of_arrays(y_test, y_pred)) 
    print("ACCURACY OF THE MODEL* (train): Fold number "+str(counter)+" :", score_list_of_arrays(y_train, y_train_pred)) 
    list.append(score)
    counter+=1
print()
print("ACCURACY", list)

# y_pred = clf.predict(X_test)
# y_train_pred = clf.predict(X_train) 
t2= datetime.now()
# print("ACCURACY OF THE MODEL (test):", metrics.accuracy_score(y_test, y_pred)) 
# print("ACCURACY OF THE MODEL (train):", metrics.accuracy_score(y_train, y_train_pred)) 
print("Time taken:", t2-t1)