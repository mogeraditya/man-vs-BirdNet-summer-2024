import numpy as np
import pickle
import os
import random
import support_functions_for_training as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns

from sklearn.multioutput import ClassifierChain

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

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
plot= storage+ "plots\\clf_chain_ordered_by_counts_descending\\"
make_dir(plot)
dir_heard= common_resources+ "heard_datafiles\\"
dir_seen= common_resources+ "seen_datafiles\\"

dir_bn= common_resources+ "store_embeddings_bn_dict\\"
dir_bn_stack= common_resources+ "store_embeddings_bn_stack_dict\\"
dir_vggish= common_resources+ "store_vggish_embeddings\\"
dir_vggish_stack= common_resources+ "store_vggish_embeddings_stacked\\"

loop1, labels1= [dir_seen], ["seen"]
loop2, labels2= [dir_bn, dir_bn_stack, dir_vggish, dir_vggish_stack], ["bn", "bn_stack", "vggish", "vggish_stack"]

inputs= []
clfs_set=[MultinomialNB(), GaussianNB()]
clfs_label_set= ["multinomial", "gaussian"]
for item2 in range(len(loop2)):
    dir= plot+labels1[0]+"_"+labels2[item2]+"\\"
    big_label= labels1[0]+"_"+labels2[item2]
    make_dir(dir)
    os.chdir(dir)
    x,y,groups= sf.get_x_and_y(loop2[item2], loop1[0])
    precision_recall, kind, label, precision_recall1= [],[],[],[]

    for it in range(len(clfs_set)):
        base_clf= clfs_set[it]
        print(base_clf)
        clf= ClassifierChain(base_clf, order='random', random_state=42)
        kf = KFold(n_splits=3)
        kf.get_n_splits(x, y)
        array_store_prec, array_store_rec, array_store_f1 =[], [], []
        array_store_prec1, array_store_rec1, array_store_f11 =[], [], []

        for i, (train_index, test_index) in enumerate(kf.split(x, y)):
            x_train, x_test = np.array(x)[train_index.astype(int)], np.array(x)[test_index.astype(int)]
            y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_train_pred = clf.predict(x_train) 
            score1= sf.score_list_of_arrays(y_test, y_pred)[0]
            score2= sf.score_list_of_arrays(y_test, y_pred)[1]
            score3= sf.score_list_of_arrays(y_test, y_pred)[2]

            score11= sf.score_list_of_arrays(y_train, y_train_pred)[0]
            score21= sf.score_list_of_arrays(y_train, y_train_pred)[1]
            score31= sf.score_list_of_arrays(y_train, y_train_pred)[2]

            array_store_prec.append(score1)
            array_store_rec.append(score2)
            array_store_f1.append(score3)

            array_store_prec1.append(score11)
            array_store_rec1.append(score21)
            array_store_f11.append(score31)
        
        Score1= np.mean(array_store_prec)
        Score2= np.mean(array_store_rec)
        Score3= np.mean(array_store_f1)
        print(Score1, Score2, Score3)
        Score11= np.mean(array_store_prec1)
        Score21= np.mean(array_store_rec1)
        Score31= np.mean(array_store_f11)

        precision_recall.append(Score1)
        precision_recall.append(Score2)
        precision_recall.append(Score3)

        precision_recall1.append(Score11)
        precision_recall1.append(Score21)
        precision_recall1.append(Score31)

        kind.append("precision")
        kind.append("recall")
        kind.append("f1 score")
        label.append(clfs_label_set[it])
        label.append(clfs_label_set[it])
        label.append(clfs_label_set[it])

    os.chdir(plot)
    df= pd.DataFrame(zip(label, kind, precision_recall), columns=["distribution type", "metric","freq (min 0, max 1)"])
    df1= pd.DataFrame(zip(label, kind, precision_recall1), columns=["distribution type", "metric","freq (min 0, max 1)"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="distribution type", hue="metric", y="freq (min 0, max 1)", data=df)
    plt.title(big_label+" KFold cross validation test")
    plt.ylim(0, 1)
    plt.savefig(big_label+"_clf_chain_test.png")
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="distribution type", hue="metric", y="freq (min 0, max 1)", data=df1)
    plt.title(big_label+" KFold cross validation train")
    plt.ylim(0, 1)
    plt.savefig(big_label+"_clf_chain_train.png")
    plt.clf()





        # dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
        # big_label= labels1[item1]+"_"+labels2[item2]
        # make_dir(dir)
        # os.chdir(dir)
        # clf_prec_arr_over_n_est= open_array("precision_n_est")
        # clf_rec_arr_over_n_est= open_array("precision_train_n_est")
        # clf_prec_arr_over_n_est_train= open_array("recall_n_est")
        # clf_rec_arr_over_n_est_train= open_array("recall_train_n_est")
        
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est, label="test")
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est_train, label="train")
        # plt.legend()
        # plt.title(big_label+ " knn precision over varied n_neighbours")
        # plt.ylabel("precision")
        # plt.xlabel("n_neighbours")
        # plt.ylim(0,1)
        # plt.savefig("knn_w_varied_n_neigh_prec.png")
        # plt.clf()
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est, label="test")
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est_train, label="train")
        # plt.legend()
        # plt.title(big_label+ " knn recall over varied n_neighbours")
        # plt.ylabel("recall")
        # plt.xlabel("n_neighbours")
        # plt.ylim(0,1)
        # plt.savefig("knn_w_varied_n_neigh_rec.png")
        # plt.clf()
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est, label="precision test")
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est_train, label="precision train")
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est, label="recall test")
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est_train, label="recall train")
        # plt.legend()
        # plt.title(big_label+ " knn over varied n_neighbours")
        # plt.ylabel("freq")
        # plt.xlabel("n_neighbours")
        # plt.ylim(0,1.01)
        # plt.savefig("knn_w_varied_n_neigh_both.png")
        # plt.clf()