# #trying to write the code myself

# import math
# import random
# import pandas as pd
# import numpy as np

# def encode_class(mydata):
#     classes = []
#     for i in range(len(mydata)):
#         if mydata[i][-1] not in classes:
#             classes.append(mydata[i][-1])
#     for i in range(len(classes)):
#         for j in range(len(mydata)):
#             if mydata[j][-1] == classes[i]:
#                 mydata[j][-1] = i
#     return mydata'
import numpy as np
from sklearn.naive_bayes import MultinomialNB


import os
import glob
import numpy as np
import pandas as pd

import pickle 

from statistics import mean, stdev

#input is a dict with embeddings for every minute 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

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

subset_merged_dict= {i:merged_dictionary[i] for i in list(y_dictionary.keys()) if i in merged_dictionary}
subset_y_dict= {i: y_dictionary[i] for i in list(subset_merged_dict.keys()) if i in y_dictionary}
x= list(subset_merged_dict.values())
x= [list(i) for i in x]
y= list(subset_y_dict.values())
y= [list(i) for i in y]
print(len(y), len(x))

clf = MultinomialNB()

kf = KFold(n_splits=3)
kf.get_n_splits(x)
store_info=[]
print(kf)  
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
    y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
    clf.fit(x_train, y_train)
    store_info.append(clf.score(x_test, y_test))
print(store_info)