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
plot= "D:\\github\\man-vs-BirdNet-summer-2024-\\individual_classifiers\\plots\\knn\\corr_incorr\\site\\"
plot_grouped= "D:\\github\\man-vs-BirdNet-summer-2024-\\individual_classifiers\\plots\\knn\\corr_incorr\\site\\group_bar_plot"
dir_heard= common_resources+ "heard_datafiles\\"
dir_seen= common_resources+ "seen_datafiles\\"

dir_bn= common_resources+ "store_embeddings_bn_dict\\"
dir_bn_stack= common_resources+ "store_embeddings_bn_stack_dict\\"
dir_vggish= common_resources+ "store_vggish_embeddings\\"
dir_vggish_stack= common_resources+ "store_vggish_embeddings_stacked\\"

loop1, labels1= [dir_seen], ["seen"]
loop2, labels2= [dir_bn, dir_vggish], ["bn", "vggish"]

birds= ["rose ringed parakeet", "spotted dove", "oriental magpie robin", "white cheeked barbet", "large billed crow", "red whiskered bulbul", "black kite"]
n_neigh_given_bird= [3, 3, 3, 20, 3,20,3]
plot_x, plot_y, plot_hue= [],[],[]
recall_arr, bn_vggish, y_axis= [],[], []



for item1 in range(len(loop1)):
    for item2 in range(len(loop2)):
        # general_test_correct_counts= np.zeros()
        for species_name in birds:
            neigh= n_neigh_given_bird[birds.index(species_name)]
            make_dir(plot); os.chdir(plot)
            dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
            big_label= labels1[item1]+"_"+labels2[item2]

            x, y, groups, date_groups= sf.get_xy_for_individual_classsifier(loop2[item2], loop1[item1], species_name)
            X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42); print(y_train[10])
            date_train, date_test, y_train, y_test = train_test_split(date_groups, y, stratify=y, test_size=0.3, random_state=42) ; print(y_train[10], date_train[10])

            clf, clf_label= KNeighborsClassifier(n_jobs=4, n_neighbors=neigh), "25_knn"

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_train_pred = clf.predict(X_train) 
            
            unique_dates = list(set(date_test))
            unique_dates = [int(i) for i in unique_dates]
            unique_dates= list(np.sort(unique_dates)); unique_dates = [str(i) for i in unique_dates]
            test_correct_counts, test_incorrect_counts= sf.get_corr_incorr_counts(date_test,unique_dates, y_pred, y_test)
            # unique_dates = list(set(date_train))
            # train_correct_counts, train_incorrect_counts= sf.get_corr_incorr_counts(date_train,unique_dates, y_train_pred, y_train)
            # print(train_correct_counts, train_incorrect_counts, test_correct_counts, test_incorrect_counts)
            # print(np.sum(train_correct_counts+ train_incorrect_counts+ test_correct_counts+ test_incorrect_counts)) #checking if it makes sense
            # print(len(unique_dates))
            # os.chdir(plot)
            # plt.figure(figsize=(16,5))
            
            # plt.bar(unique_dates,train_correct_counts, alpha=0.3, label= "train (n_correct="+str(int(np.sum(train_correct_counts)))+ " out of "+str(len(y_train_pred))+")")
            # plt.bar(unique_dates,test_correct_counts, alpha=0.75, label= "test (n_correct="+str(int(np.sum(test_correct_counts)))+ " out of "+str(len(y_pred))+")")
            
            # plt.title("counts of correct per date")
            # plt.legend()
            # plt.xticks(ticks=range(len(unique_dates)), labels= unique_dates, rotation= 90)
            # plt.tight_layout()
            # plt.savefig(species_name+"_correct_counts_hist_"+big_label+".png")
            # plt.clf()
            
            # plt.figure(figsize=(16,5))
            
            # plt.bar(unique_dates, train_incorrect_counts, alpha=0.3, label= "train (n_incorrect="+str(int(np.sum(train_incorrect_counts)))+ " out of "+str(len(y_train_pred))+")")
            # plt.bar(unique_dates, test_incorrect_counts, alpha=0.75, label= "test (n_incorrect="+str(int(np.sum(test_incorrect_counts)))+ " out of "+str(len(y_pred))+")")
            
            # plt.title("counts of incorrect per date")
            # plt.legend()
            # plt.xticks(ticks=range(len(unique_dates)), labels= unique_dates, rotation= 90)
            # plt.tight_layout()
            # plt.savefig(species_name+"_incorrect_counts_hist_"+big_label+".png")
            # plt.clf()
            
            make_dir(plot_grouped)
            os.chdir(plot_grouped)
            df_grouped_plot= pd.DataFrame(columns= ["date", "type", "count"])
            consolidated_counts= list(test_correct_counts)+ list(test_incorrect_counts)
            type_array= ["correct" for i in test_correct_counts] + ["incorrect" for i in test_incorrect_counts]
            date_array= unique_dates+unique_dates
            df_grouped_plot["date"]= date_array; df_grouped_plot["type"]= type_array; df_grouped_plot["count"]= consolidated_counts
            #plot
            plt.figure(figsize=(16,5))
            sns.barplot(data= df_grouped_plot, x="date", y="count", hue= "type")
            plt.title(f"counts of correct/ incorrect per date; n_neigh={neigh}")
            plt.legend()
            plt.xticks(ticks=range(len(unique_dates)), labels= unique_dates, rotation= 90)
            plt.tight_layout()
            plt.savefig(species_name+"_counts_hist_"+big_label+".png")
            plt.clf()
            
            
            
            
# for item1 in range(len(loop1)):
#     for item2 in range(len(loop2)):
#         # general_test_correct_counts= np.zeros()
#         for species_name in birds:
#             neigh= n_neigh_given_bird[birds.index(species_name)]
#             make_dir(plot); os.chdir(plot)
#             dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
#             big_label= labels1[item1]+"_"+labels2[item2]

#             x, y, groups, date_groups= sf.get_xy_for_individual_classsifier(loop2[item2], loop1[item1], species_name)
#             X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42); print(y_train[10])
#             date_train, date_test, y_train, y_test = train_test_split(date_groups, y, stratify=y, test_size=0.3, random_state=42) ; print(y_train[10], date_train[10])

#             clf, clf_label= KNeighborsClassifier(n_jobs=4, n_neighbors=neigh), "25_knn"

#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_test)
#             y_train_pred = clf.predict(X_train) 
#             unique = list(set(groups))
#             test_correct_counts, test_incorrect_counts= sf.get_corr_incorr_counts(groups,unique, y_pred, y_test)
#             train_correct_counts, train_incorrect_counts= sf.get_corr_incorr_counts(groups,unique, y_train_pred, y_train)
#             # print(train_correct_counts, train_incorrect_counts, test_correct_counts, test_incorrect_counts)
#             # print(np.sum(train_correct_counts+ train_incorrect_counts+ test_correct_counts+ test_incorrect_counts)) #checking if it makes sense
#             print(len(unique))
#             os.chdir(plot)
#             plt.figure(figsize=(16,5))
            
#             plt.bar(unique, train_correct_counts, alpha=0.3, label= "train (n_correct="+str(int(np.sum(train_correct_counts)))+ " out of "+str(len(y_train_pred))+")")
#             plt.bar(unique, test_correct_counts, alpha=0.75, label= "test (n_correct="+str(int(np.sum(test_correct_counts)))+ " out of "+str(len(y_pred))+")")
            
#             plt.title("counts of correct per site")
#             plt.legend()
#             plt.xticks(ticks=range(len(unique)), labels= unique, rotation= 90)
#             plt.tight_layout()
#             plt.savefig(species_name+"_correct_counts_hist_"+big_label+".png")
#             plt.clf()
            
#             plt.figure(figsize=(16,5))
            
#             plt.bar(unique, train_incorrect_counts, alpha=0.3, label= "train (n_incorrect="+str(int(np.sum(train_incorrect_counts)))+ " out of "+str(len(y_train_pred))+")")
#             plt.bar(unique, test_incorrect_counts, alpha=0.75, label= "test (n_incorrect="+str(int(np.sum(test_incorrect_counts)))+ " out of "+str(len(y_pred))+")")
            
#             plt.title("counts of incorrect per site")
#             plt.legend()
#             plt.xticks(ticks=range(len(unique)), labels= unique, rotation= 90)
#             plt.tight_layout()
#             plt.savefig(species_name+"_incorrect_counts_hist_"+big_label+".png")
#             plt.clf()
            
#             make_dir(plot_grouped)
#             os.chdir(plot_grouped)
#             df_grouped_plot= pd.DataFrame(columns= ["site", "type", "count"])
#             consolidated_counts= list(test_correct_counts)+ list(test_incorrect_counts)
#             type_array= ["correct" for i in test_correct_counts] + ["incorrect" for i in test_incorrect_counts]
#             site_array= unique+unique
#             df_grouped_plot["site"]= site_array; df_grouped_plot["type"]= type_array; df_grouped_plot["count"]= consolidated_counts
#             #plot
#             plt.figure(figsize=(16,5))
#             sns.barplot(data= df_grouped_plot, x="site", y="count", hue= "type")
#             plt.title(f"counts of correct/ incorrect per site; n_neigh={neigh}")
#             plt.legend()
#             plt.xticks(ticks=range(len(unique)), labels= unique, rotation= 90)
#             plt.tight_layout()
#             plt.savefig(species_name+"_counts_hist_"+big_label+".png")
#             plt.clf()
                
            