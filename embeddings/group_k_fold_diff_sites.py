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
import glob

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
plot= storage+ "plots\\only_seen_data\\grouped_by_sites\\"
dir_heard= common_resources+ "heard_datafiles\\"
dir_seen= common_resources+ "seen_datafiles\\"

dir_bn= common_resources+ "store_embeddings_bn_dict\\"
dir_bn_stack= common_resources+ "store_embeddings_bn_stack_dict\\"
dir_vggish= common_resources+ "store_vggish_embeddings\\"
dir_vggish_stack= common_resources+ "store_vggish_embeddings_stacked\\"

loop1, labels1= [dir_seen], ["heard"]
loop2, labels2= [dir_bn, dir_vggish], ["bn", "vggish"]

n_splits= 3

precision_recall, kind, label, precision_recall1= [],[],[],[]
for item1 in range(len(loop1)):
    precision_recall, kind, label, precision_recall1= [],[],[],[]
    for item2 in range(len(loop2)):
        dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
        big_label= labels1[item1]+"_"+labels2[item2]
        make_dir(dir)
        os.chdir(dir)
        x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
        gkf = GroupKFold(n_splits= n_splits)
        gkf.get_n_splits(x, y)
        clf= KNeighborsClassifier(n_jobs=4, n_neighbors=11)
        tp_arr, fp_arr, fn_arr= [], [], []
        tp_tr_arr, fp_tr_arr, fn_tr_arr= [], [], []
        for i, (train_index, test_index) in enumerate(gkf.split(x, y, groups)):
            x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
            y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_train_pred = clf.predict(x_train) 

            tp, fp, fn= sf.score_list_of_arrays(y_test, y_pred)
            tp_tr, fp_tr, fn_tr= sf.score_list_of_arrays(y_train, y_train_pred)

            tp_arr.append(tp)
            fp_arr.append(fp)
            fn_arr.append(fn)

            tp_tr_arr.append(tp_tr)
            fp_tr_arr.append(fp_tr)
            fn_tr_arr.append(fn_tr)
        
        Score1= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fp_arr))
        Score2= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fn_arr))
        Score3= (2*Score1*Score2)/(Score1+Score2)

        Score11= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fp_tr_arr))
        Score21= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fn_tr_arr))
        Score31= (2*Score11*Score21)/(Score11+Score21)
        print(Score1, Score11)
        precision_recall.append(Score1)
        precision_recall.append(Score2)
        precision_recall.append(Score3)

        precision_recall1.append(Score11)
        precision_recall1.append(Score21)
        precision_recall1.append(Score31)

        kind.append("precision")
        kind.append("recall")
        kind.append("f1 score")
        label.append(labels2[item2])
        label.append(labels2[item2])
        label.append(labels2[item2])
    make_dir(plot)
    os.chdir(plot)
    df= pd.DataFrame(zip(label, kind, precision_recall), columns=["embedding type", "metric","freq (min 0, max 1)"])
    df1= pd.DataFrame(zip(label, kind, precision_recall1), columns=["embedding type", "metric","freq (min 0, max 1)"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df)
    plt.title(labels1[item1]+ " GroupKFold (site) cross validation test n_splits_"+str(n_splits))
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(labels1[item1]+"_gkfold_knn_n12_test_seen_only_sites_n"+str(n_splits)+".png")
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df1)
    plt.title(labels1[item1]+" GroupKFold (sites) cross validation train n_splits_"+str(n_splits))
    plt.ylim(0, 1)
    plt.savefig(labels1[item1]+"_gkfold_knn_n12_train_seen_only_sites_n"+str(n_splits)+".png")
    plt.clf()

precision_recall, kind, label, precision_recall1= [],[],[],[]
for item1 in range(len(loop1)):
    precision_recall, kind, label, precision_recall1= [],[],[],[]
    for item2 in range(len(loop2)):
        dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
        big_label= labels1[item1]+"_"+labels2[item2]
        make_dir(dir)
        os.chdir(dir)
        x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
        kf = KFold(n_splits= n_splits)
        kf.get_n_splits(x, y)
        clf= KNeighborsClassifier(n_jobs=4, n_neighbors=11)
        array_store_prec, array_store_rec, array_store_f1 =[], [], []
        array_store_prec1, array_store_rec1, array_store_f11 =[], [], []
        for i, (train_index, test_index) in enumerate(kf.split(x, y)):
            x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
            y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_train_pred = clf.predict(x_train) 
            tp, fp, fn= sf.score_list_of_arrays(y_test, y_pred)
            tp_tr, fp_tr, fn_tr= sf.score_list_of_arrays(y_train, y_train_pred)

            tp_arr.append(tp)
            fp_arr.append(fp)
            fn_arr.append(fn)

            tp_tr_arr.append(tp_tr)
            fp_tr_arr.append(fp_tr)
            fn_tr_arr.append(fn_tr)
        
        Score1= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fp_arr))
        Score2= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fn_arr))
        Score3= (2*Score1*Score2)/(Score1+Score2)

        Score11= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fp_tr_arr))
        Score21= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fn_tr_arr))
        Score31= (2*Score11*Score21)/(Score11+Score21)
        print(Score1, Score11)
        precision_recall.append(Score1)
        precision_recall.append(Score2)
        precision_recall.append(Score3)

        precision_recall1.append(Score11)
        precision_recall1.append(Score21)
        precision_recall1.append(Score31)

        kind.append("precision")
        kind.append("recall")
        kind.append("f1 score")
        label.append(labels2[item2])
        label.append(labels2[item2])
        label.append(labels2[item2])

    os.chdir(plot)
    df= pd.DataFrame(zip(label, kind, precision_recall), columns=["embedding type", "metric","freq (min 0, max 1)"])
    df1= pd.DataFrame(zip(label, kind, precision_recall1), columns=["embedding type", "metric","freq (min 0, max 1)"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df)
    plt.title(labels1[item1]+" KFold (sites) cross validation test n_splits_"+str(n_splits))
    plt.ylim(0, 1)
    plt.savefig(labels1[item1]+"_kfold_knn_n12_test_seen_only_sites_n"+str(n_splits)+".png")
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df1)
    plt.title(labels1[item1]+" KFold (sites) cross validation train n_splits_"+str(n_splits))
    plt.ylim(0, 1)
    plt.savefig(labels1[item1]+"_kfold_knn_n12_train_seen_only_sites_n"+str(n_splits)+".png")
    plt.clf()

# precision_recall, kind, label, precision_recall1= [],[],[],[]
# for item1 in range(len(loop1)):
#     precision_recall, kind, label, precision_recall1= [],[],[],[]
#     for item2 in range(len(loop2)):
#         dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
#         big_label= labels1[item1]+"_"+labels2[item2]
#         make_dir(dir)
#         os.chdir(dir)
#         x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
#         gkf = GroupKFold(n_splits= n_splits)
#         gkf.get_n_splits(x, y)
#         clf= KNeighborsClassifier(n_jobs=4, n_neighbors=11)
#         tp_arr, fp_arr, fn_arr= [], [], []
#         tp_tr_arr, fp_tr_arr, fn_tr_arr= [], [], []
#         for i, (train_index, test_index) in enumerate(gkf.split(x, y, groups)):
#             x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
#             y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
#             clf.fit(x_train, y_train)
#             y_pred = clf.predict(x_test)
#             y_train_pred = clf.predict(x_train) 

#             tp, fp, fn= sf.score_list_of_arrays(y_test, y_pred)
#             tp_tr, fp_tr, fn_tr= sf.score_list_of_arrays(y_train, y_train_pred)

#             tp_arr.append(tp)
#             fp_arr.append(fp)
#             fn_arr.append(fn)

#             tp_tr_arr.append(tp_tr)
#             fp_tr_arr.append(fp_tr)
#             fn_tr_arr.append(fn_tr)
        
#         Score1= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fp_arr))
#         Score2= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fn_arr))
#         Score3= (2*Score1*Score2)/(Score1+Score2)

#         Score11= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fp_tr_arr))
#         Score21= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fn_tr_arr))
#         Score31= (2*Score11*Score21)/(Score11+Score21)
#         print(Score1, Score11)
#         precision_recall.append(Score1)
#         precision_recall.append(Score2)
#         precision_recall.append(Score3)

#         precision_recall1.append(Score11)
#         precision_recall1.append(Score21)
#         precision_recall1.append(Score31)

#         kind.append("precision")
#         kind.append("recall")
#         kind.append("f1 score")
#         label.append(labels2[item2])
#         label.append(labels2[item2])
#         label.append(labels2[item2])
#     make_dir(plot)
#     os.chdir(plot)
#     df= pd.DataFrame(zip(label, kind, precision_recall), columns=["embedding type", "metric","freq (min 0, max 1)"])
#     df1= pd.DataFrame(zip(label, kind, precision_recall1), columns=["embedding type", "metric","freq (min 0, max 1)"])
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df)
#     plt.title(labels1[item1]+ " GroupKFold (site) cross validation test n_splits_"+str(n_splits))
#     plt.ylim(0, 1)
#     # plt.show()
#     plt.savefig(labels1[item1]+"_gkfold_knn_n12_test_seen_only_sites_n"+str(n_splits)+".png")
#     plt.clf()
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df1)
#     plt.title(labels1[item1]+" GroupKFold (sites) cross validation train n_splits_"+str(n_splits))
#     plt.ylim(0, 1)
#     plt.savefig(labels1[item1]+"_gkfold_knn_n12_train_seen_only_sites_n"+str(n_splits)+".png")
#     plt.clf()

# precision_recall, kind, label, precision_recall1= [],[],[],[]
# for item1 in range(len(loop1)):
#     precision_recall, kind, label, precision_recall1= [],[],[],[]
#     for item2 in range(len(loop2)):
#         dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
#         big_label= labels1[item1]+"_"+labels2[item2]
#         make_dir(dir)
#         os.chdir(dir)
#         x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
#         kf = KFold(n_splits= n_splits)
#         kf.get_n_splits(x, y)
#         clf= KNeighborsClassifier(n_jobs=4, n_neighbors=11)
#         array_store_prec, array_store_rec, array_store_f1 =[], [], []
#         array_store_prec1, array_store_rec1, array_store_f11 =[], [], []
#         for i, (train_index, test_index) in enumerate(kf.split(x, y)):
#             x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
#             y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
#             clf.fit(x_train, y_train)
#             y_pred = clf.predict(x_test)
#             y_train_pred = clf.predict(x_train) 
#             tp, fp, fn= sf.score_list_of_arrays(y_test, y_pred)
#             tp_tr, fp_tr, fn_tr= sf.score_list_of_arrays(y_train, y_train_pred)

#             tp_arr.append(tp)
#             fp_arr.append(fp)
#             fn_arr.append(fn)

#             tp_tr_arr.append(tp_tr)
#             fp_tr_arr.append(fp_tr)
#             fn_tr_arr.append(fn_tr)
        
#         Score1= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fp_arr))
#         Score2= np.sum(tp_arr)/ (np.sum(tp_arr)+ np.sum(fn_arr))
#         Score3= (2*Score1*Score2)/(Score1+Score2)

#         Score11= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fp_tr_arr))
#         Score21= np.sum(tp_tr_arr)/ (np.sum(tp_tr_arr)+ np.sum(fn_tr_arr))
#         Score31= (2*Score11*Score21)/(Score11+Score21)
#         print(Score1, Score11)
#         precision_recall.append(Score1)
#         precision_recall.append(Score2)
#         precision_recall.append(Score3)

#         precision_recall1.append(Score11)
#         precision_recall1.append(Score21)
#         precision_recall1.append(Score31)

#         kind.append("precision")
#         kind.append("recall")
#         kind.append("f1 score")
#         label.append(labels2[item2])
#         label.append(labels2[item2])
#         label.append(labels2[item2])

#     os.chdir(plot)
#     df= pd.DataFrame(zip(label, kind, precision_recall), columns=["embedding type", "metric","freq (min 0, max 1)"])
#     df1= pd.DataFrame(zip(label, kind, precision_recall1), columns=["embedding type", "metric","freq (min 0, max 1)"])
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df)
#     plt.title(labels1[item1]+" KFold (sites) cross validation test n_splits_"+str(n_splits))
#     plt.ylim(0, 1)
#     plt.savefig(labels1[item1]+"_kfold_knn_n12_test_seen_only_sites_n"+str(n_splits)+".png")
#     plt.clf()
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x="embedding type", hue="metric", y="freq (min 0, max 1)", data=df1)
#     plt.title(labels1[item1]+" KFold (sites) cross validation train n_splits_"+str(n_splits))
#     plt.ylim(0, 1)
#     plt.savefig(labels1[item1]+"_kfold_knn_n12_train_seen_only_sites_n"+str(n_splits)+".png")
#     plt.clf()