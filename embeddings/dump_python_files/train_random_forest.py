import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import pickle 

from datetime import datetime
from statistics import mean, stdev

#input is a dict with embeddings for every minute 

from sklearn.model_selection import train_test_split

from sklearn import metrics 


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
dir= common_resources+ "store_embeddings_dict\\"
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
with open("loclabels.pickle", 'rb') as f:
    labels_dictionary = pickle.load(f)
# print(y_dictionary)
# print(merged_dictionary)

subset_merged_dict= {i:merged_dictionary[i] for i in list(y_dictionary.keys()) if i in merged_dictionary}
subset_y_dict= {i: y_dictionary[i] for i in list(subset_merged_dict.keys()) if i in y_dictionary}
subset_labels_dict= {i:labels_dictionary[i] for i in list(subset_merged_dict.keys()) if i in labels_dictionary}
x= list(subset_merged_dict.values())
x= [list(i) for i in x]
y= list(subset_y_dict.values())
y= [list(i) for i in y]
groups= list(subset_labels_dict.values())
# groups= [list(i) for i in x]
# print(len(y), len(x), len(groups))
# print(groups)

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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

t1= datetime.now()
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_jobs=4)
# clf.fit(X_train, y_train)
 
# os.chdir(common_resources)
# with open('clf_subset_of_data.pickle', 'wb') as f:
#     pickle.dump(clf, f)
# with open('clf_subset_of_data.pickle', 'rb') as f:
#    clf = pickle.load(f)
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=3, shuffle=True, random_state=42)
# kf.get_n_splits(x)

# print(kf)  
# list= []
# counter=0
# for train_index, test_index in kf.split(x):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
#     y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     y_train_pred = clf.predict(x_train) 
#     # print("ACCURACY OF THE MODEL (test):", metrics.accuracy_score(y_test, y_pred)) 
#     # print("ACCURACY OF THE MODEL (train):", metrics.accuracy_score(y_train, y_train_pred)) 
#     score= score_list_of_arrays(y_test, y_pred)
#     print("ACCURACY OF THE MODEL* (test): Fold number "+str(counter)+" :", score_list_of_arrays(y_test, y_pred)) 
#     print("ACCURACY OF THE MODEL* (train): Fold number "+str(counter)+" :", score_list_of_arrays(y_train, y_train_pred)) 
#     list.append(score)
#     counter+=1
# print()
# print("ACCURACY", list)
import matplotlib.pyplot as plt
list_accuracy= []
list_accuracy_train=[]
wcss=[]
for i in range(1,20):
    clf = KNeighborsClassifier(n_jobs=4, n_neighbors=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train) 
    t2= datetime.now()
    print("ACCURACY OF THE MODEL (test):", metrics.accuracy_score(y_test, y_pred)) 
    print("ACCURACY OF THE MODEL (train):", metrics.accuracy_score(y_train, y_train_pred)) 
    print("Time taken:", t2-t1)
    # print("ACCURACY OF THE MODEL* (test):", score_list_of_arrays(y_test, y_pred)) 
    # print("ACCURACY OF THE MODEL* (train):", score_list_of_arrays(y_train, y_train_pred)) 
    list_accuracy.append(metrics.accuracy_score(y_test, y_pred))
    list_accuracy_train.append(metrics.accuracy_score(y_train, y_train_pred))

plt.plot(np.arange(1,20),list_accuracy, label="test")
plt.plot(np.arange(1,20),list_accuracy_train, label="train")
plt.legend()
plt.title("accuracies with varied n_neighbours")
plt.ylabel("accuracy")
plt.xlabel("n_neigbours")
# plt.ylim(0,0.5)
plt.show()
# print(y_pred[0])
# print(y_test[0])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }
# rfc= RandomForestClassifier(n_jobs=4)
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(x_train, y_train)
# print(CV_rfc.best_params_)


  
# Feature Scaling for input features.
# scaler = preprocessing.MinMaxScaler()
# x_scaled = scaler.fit_transform(x)
  

# lr = RandomForestClassifier(max_depth=2, n_estimators=3000,
#     min_samples_split=3, max_leaf_nodes=5,
#     random_state=22)
# # # skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
# lst_accu_stratified = []
  
# for train_index, test_index in skf.split(x, y):
#     x_train_fold, x_test_fold = x[train_index], x[test_index]
#     y_train_fold, y_test_fold = y[train_index], y[test_index]
#     lr.fit(x_train_fold, y_train_fold)
#     lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold))

# # # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# # # y = np.array([1, 2, 3, 4])
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=3)
# kf.get_n_splits(x)

# print(kf)  

# for train_index, test_index in kf.split(x):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
#     y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
#     lr.fit(x_train, y_train)
#     lst_accu_stratified.append(lr.score(x_test, y_test))

# # print(lst_accu_stratified)
# from sklearn.model_selection import GroupKFold



# gkf = GroupKFold(n_splits=3)
# gkf.get_n_splits(x, y)
# print(gkf.split(x, y, groups))
# for i, (train_index, test_index) in enumerate(gkf.split(x, y, groups)):
#     x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
#     y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
#     lr.fit(x_train, y_train)
    

# print(lst_accu_stratified)

# from sklearn.model_selection import permutation_test_score

# score, permutation_scores, pvalue = permutation_test_score(
#     lr, x, y, random_state=42, n_permutations=20, n_jobs=4
# )
# print(f"Original Score: {score:.3f}")
# print(
#     f"Permutation Scores: {permutation_scores.mean():.3f} +/- "
#     f"{permutation_scores.std():.3f}"
# )

# import matplotlib.pyplot as plt
# plt.hist(permutation_scores)
# plt.title("hist of permutations")
# plt.show()