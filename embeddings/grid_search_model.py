import numpy as np
import pickle
import os
import random
import support_functions_for_training as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
storage= "d:\\Research\\analyze_embeddings\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\"
plot= storage+ "plots\\"
dir_heard= common_resources+ "heard_datafiles\\"
dir_seen= common_resources+ "seen_datafiles\\"

dir_bn= common_resources+ "store_embeddings_bn_dict\\"
dir_bn_stack= common_resources+ "store_embeddings_bn_stack_dict\\"
dir_vggish= common_resources+ "store_vggish_embeddings\\"
dir_vggish_stack= common_resources+ "store_vggish_embeddings_stacked\\"

loop1, labels1= [dir_seen], ["seen"]
loop2, labels2= [dir_bn, dir_bn_stack, dir_vggish, dir_vggish_stack], ["bn", "bn_stack", "vggish", "vggish_stack"]
def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)
def merge_lists(list_of_lists):
    null= []
    for list in list_of_lists:
        for item in list:
            null.append(item)
    return null
def save_array(filename, array):
    array= np.array(array)
    with open(filename+'.npy', 'wb') as f:
        np.save(f, array)

from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

for item1 in range(len(loop1)):
    for item2 in range(len(loop2)):
        dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
        big_label= labels1[item1]+"_"+labels2[item2]
        make_dir(dir)
        os.chdir(dir)
        x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        param_grid = { 
            'n_neighbors': [4,20,2],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1,2],
            "leaf_size": [26,28,30,32,34]
        }
        clf= KNeighborsClassifier(n_jobs=4)
        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3, scoring=sf.scorer)
        CV_rfc.fit(x_train, y_train)
        with open('grid_search_results.pickle', 'wb') as handle:
            dict_to_save= CV_rfc.cv_results_
        df= pd.concat([pd.DataFrame(clf.cv_results_["params"]),pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
        df.to_csv("datafram_grid_search.csv")
        print(CV_rfc.best_params_)