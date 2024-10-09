import numpy as np
import pickle
import os
import random
import support_functions_for_training as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import recall_score, precision_score, f1_score

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
def open_array(filename):
    with open(filename+'.npy', 'rb') as f:
        array= np.load(f)
    return array

storage= "d:\\Research\\analyze_embeddings\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\"
plot= "D:\\github\\man-vs-BirdNet-summer-2024-\\individual_classifiers\\plots\\knn\\general_stats\\"
dir_heard= common_resources+ "heard_datafiles\\"
dir_seen= common_resources+ "seen_datafiles\\"

dir_bn= common_resources+ "store_embeddings_bn_dict\\"
dir_bn_stack= common_resources+ "store_embeddings_bn_stack_dict\\"
dir_vggish= common_resources+ "store_vggish_embeddings\\"
dir_vggish_stack= common_resources+ "store_vggish_embeddings_stacked\\"

loop1, labels1= [dir_seen], ["seen"]
loop2, labels2= [dir_bn, dir_vggish], ["bn", "vggish"]

birds= ["rose ringed parakeet", "spotted dove", "oriental magpie robin", "white cheeked barbet", "large billed crow", "red whiskered bulbul", "black kite"]
plot_x, plot_y, plot_hue= [],[],[]
recall_arr, bn_vggish, y_axis= [],[], []



for item1 in range(len(loop1)):
    for item2 in range(len(loop2)):
        for species_name in birds:
            dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
            big_label= labels1[item1]+"_"+labels2[item2]
            make_dir(dir)
            os.chdir(dir)
            
            x, y, groups, date_groups= sf.get_xy_for_individual_classsifier(loop2[item2], loop1[item1], species_name)

            kf = KFold(n_splits=3)
            kf.get_n_splits(x, y)
            # clf, clf_label= KNeighborsClassifier(n_jobs=4, n_neighbors=3), "knn"
            clf, clf_label= KNeighborsClassifier(n_jobs=4, n_neighbors=25), "25_knn"
            # clf, clf_label= LogisticRegression(), "linear"
            # clf, clf_label= RidgeClassifier(alpha= 1.0), "ridge"
            # clf, clf_label= RidgeClassifier(alpha= 0.5), "ridge_0.5"
            # X_test, Y_test, X_train, Y_train, Pred, Pred_train= [], [], [], [], [], []
            cm= np.zeros(shape=(2,2))
            cm_train= np.zeros(shape=(2,2))
            for i, (train_index, test_index) in enumerate(kf.split(x, y)):
                x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
                y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                y_train_pred = clf.predict(x_train) 
                cm_update = confusion_matrix(y_test, y_pred)
                cm_update_train= confusion_matrix(y_train, y_train_pred)
                cm+=cm_update
                cm_train+=cm_update_train
                
                
            target_names= [0,1]
            cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cmn_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, vmin=0.0, vmax=1.0)
            os.chdir(dir)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title(species_name+" "+clf_label+" test confusion_matrix")
            plt.savefig(species_name+"_"+clf_label+"_test_confusion_matrix.png")
            plt.clf()  
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(cmn_train, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, vmin=0.0, vmax=1.0)
            os.chdir(dir)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title(species_name+" "+clf_label+" train confusion_matrix")
            plt.savefig(species_name+"_"+clf_label+"_train_confusion_matrix.png")     
    #         recall= recall_score(y_test, y_pred)
    #         precision= precision_score(y_test, y_pred)
    #         plot_y.append(recall) 
    #         plot_y.append(precision)
    #         plot_hue.append("recall")
    #         plot_hue.append("precision")
    #         plot_x.append(species_name)
    #         plot_x.append(species_name)
            
    #         bn_vggish.append(labels2[item2])
    #         recall_arr.append(precision)
    #         y_axis.append(species_name)

    #     # plt.figure(figsize=(15,9))    
    #     os.chdir(plot)       
    #     sns.barplot(x= plot_x, y=plot_y, hue= plot_hue, errorbar=None)
    #     plt.ylim(0,1.01)
    #     plt.xticks(rotation=90, ha='right')
    #     plt.title("individual model across species prec/rec")
    #     plt.tight_layout()
    #     plt.savefig(labels2[item2]+ "precision_recall.png")
    #     plt.clf()
        
    # os.chdir(plot)       
    # sns.barplot(x=y_axis , y=recall_arr, hue= bn_vggish, errorbar=None)
    # plt.ylim(0,1.01)
    # plt.xticks(rotation=90, ha='right')
    # plt.title("individual model across species bn/vggish for precision")
    # plt.tight_layout()
    # plt.savefig(labels1[item1]+ "_bn_vs_vggish_prec.png")
    
            

        
