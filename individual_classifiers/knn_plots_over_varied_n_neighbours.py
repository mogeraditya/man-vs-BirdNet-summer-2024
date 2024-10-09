import numpy as np
import pickle
import os
import random
import support_functions_for_training as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score

storage= "d:\\Research\\analyze_embeddings\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\"
plot= "D:\\github\\man-vs-BirdNet-summer-2024-\\individual_classifiers\\plots\\knn\\varied_n_neigh\\"
dir_heard= common_resources+ "heard_datafiles\\"
dir_seen= common_resources+ "seen_datafiles\\"

dir_bn= common_resources+ "store_embeddings_bn_dict\\"
dir_bn_stack= common_resources+ "store_embeddings_bn_stack_dict\\"
dir_vggish= common_resources+ "store_vggish_embeddings\\"
dir_vggish_stack= common_resources+ "store_vggish_embeddings_stacked\\"

loop1, labels1= [dir_seen], ["seen"]
loop2, labels2= [dir_bn, dir_vggish], ["bn", "vggish"]

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

birds= ["rose ringed parakeet", "spotted dove", "oriental magpie robin", "white cheeked barbet", "large billed crow", "red whiskered bulbul", "black kite"]

for item1 in range(len(loop1)):
    for item2 in range(len(loop2)):
        for species_name in birds:
            dir= plot+labels1[item1]+"_"+labels2[item2]+"_"+species_name+"\\"
            big_label= labels1[item1]+"_"+labels2[item2]
            make_dir(dir)
            os.chdir(dir)
            
            x, y, groups, date_groups= sf.get_xy_for_individual_classsifier(loop2[item2], loop1[item1], species_name)
            X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)
            
            clf_prec_arr_over_n_est= []
            clf_rec_arr_over_n_est= []
            clf_prec_arr_over_n_est_train= []
            clf_rec_arr_over_n_est_train= []
            for i in range(1, 40, 1):
                clf= KNeighborsClassifier(n_jobs=4, n_neighbors=i)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_train_pred = clf.predict(X_train) 
                recall= recall_score(y_test, y_pred)
                precision= precision_score(y_test, y_pred)
                score= [precision_score(y_test, y_pred), recall_score(y_test, y_pred)] #sf.score_list_of_arrays(y_test, y_pred)
                score2= [precision_score(y_train, y_train_pred), recall_score(y_train, y_train_pred)]
                print(score, score2)
                clf_prec_arr_over_n_est.append(score[0])
                clf_prec_arr_over_n_est_train.append(score2[0])
                clf_rec_arr_over_n_est.append(score[1])
                clf_rec_arr_over_n_est_train.append(score2[1])
            make_dir(plot); make_dir(dir)  
            os.chdir(dir)
            save_array(species_name+"_precision_n_est", clf_prec_arr_over_n_est)
            save_array(species_name+"_precision_train_n_est", clf_prec_arr_over_n_est_train)
            save_array(species_name+"_recall_n_est", clf_rec_arr_over_n_est)
            save_array(species_name+"_recall_train_n_est", clf_rec_arr_over_n_est_train)
            
            os.chdir(plot)
            plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est, label="precision test")
            plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est_train, label="precision train")
            plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est, label="recall test")
            plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est_train, label="recall train")
            plt.legend()
            plt.title(big_label+ " knn over varied n_neighbours for "+ species_name)
            plt.ylabel("freq")
            plt.xlabel("n_neighbours")
            plt.ylim(0,1.01)
            plt.savefig(big_label+"_knn_w_varied_n_neigh_both_"+species_name+".png")
            plt.clf()