import numpy as np
import pickle
import os
import random
import support_functions_for_training as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

loop1, labels1= [dir_heard, dir_seen], ["heard", "seen"]
loop2, labels2= [ dir_bn_stack, dir_vggish, dir_vggish_stack, dir_bn], ["bn", "bn_stack", "vggish", "vggish_stack"]
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

from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics

for item1 in range(len(loop1)):
    for item2 in range(len(loop2)):
        dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
        make_dir(dir)
        os.chdir(dir)
        x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # clf_prec_arr_over_n_est= []
        # clf_rec_arr_over_n_est= []
        # clf_prec_arr_over_n_est_train= []
        # clf_rec_arr_over_n_est_train= []
        # for i in range(1, 40, 1):
        #     clf= KNeighborsClassifier(n_jobs=4, n_neighbors=i)
        #     clf.fit(X_train, y_train)
        #     y_pred = clf.predict(X_test)
        #     y_train_pred = clf.predict(X_train) 
        #     score= sf.score_list_of_arrays(y_test, y_pred)
        #     score2= sf.score_list_of_arrays(y_train, y_train_pred)
        #     print(score, score2)
        #     clf_prec_arr_over_n_est.append(score[0])
        #     clf_prec_arr_over_n_est_train.append(score2[0])
        #     clf_rec_arr_over_n_est.append(score[1])
        #     clf_rec_arr_over_n_est_train.append(score2[1])
        # os.chdir(dir)
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est, label="test")
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est_train, label="train")
        # plt.legend()
        # plt.title("knn precision over varied n_neighbours")
        # plt.ylabel("precision")
        # plt.xlabel("n_neighbours")
        # plt.savefig("knn_w_varied_n_neigh_prec.png")
        # plt.clf()
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est, label="test")
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est_train, label="train")
        # plt.legend()
        # plt.title("knn recall over varied n_neighbours")
        # plt.ylabel("recall")
        # plt.xlabel("n_neighbours")
        # plt.savefig("knn_w_varied_n_neigh_rec.png")
        # plt.clf()
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est, label="precision test")
        # plt.plot(np.arange(1, 40, 1),clf_prec_arr_over_n_est_train, label="precision train")
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est, label="recall test")
        # plt.plot(np.arange(1, 40, 1),clf_rec_arr_over_n_est_train, label="recall train")
        # plt.legend()
        # plt.title("knn accuracy over varied n_neighbours")
        # plt.ylabel("freq")
        # plt.xlabel("n_neighbours")
        # plt.savefig("knn_w_varied_n_neigh_both.png")
        # plt.clf()

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        clf= KNeighborsClassifier(n_jobs=4, n_neighbors=12)
        clf.fit(X_train, y_train)
        with open('clf'+labels1[item1]+"_"+labels2[item2]+'.pickle', 'wb') as f:
            pickle.dump(clf, f)
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train) 
        score= sf.score_list_of_arrays(y_test, y_pred)[0]
        score2= sf.score_list_of_arrays(y_test, y_pred)[1]
        score3= sf.score_list_of_arrays(y_test, y_pred)[2]
        # score2= sf.score_list_of_arrays(y_train, y_train_pred)

        permutation_scores= []
        permutation_scores2= []
        permutation_scores3= []
        for i in range(0,1000):
            x_new= x.copy()
            random.shuffle(x_new)
            X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.3, random_state=42)
            clf= KNeighborsClassifier(n_jobs=4, n_neighbors=12)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_train_pred = clf.predict(X_train) 
            perm_score= sf.score_list_of_arrays(y_test, y_pred)[0]
            perm_score2= sf.score_list_of_arrays(y_test, y_pred)[1]
            perm_score3= sf.score_list_of_arrays(y_test, y_pred)[2]
            permutation_scores.append(perm_score)
            permutation_scores2.append(perm_score2)
            permutation_scores3.append(perm_score3)

        permutation_scores=np.array(permutation_scores)
        permutation_scores2=np.array(permutation_scores2)
        permutation_scores3=np.array(permutation_scores3)

        print(f"Original Score: {score:.3f}")
        print(
            f"Permutation Scores prec: {permutation_scores.mean():.3f} +/- "
            f"{permutation_scores.std():.3f}"
        )
        print(
            f"Permutation Scores rec: {permutation_scores2.mean():.3f} +/- "
            f"{permutation_scores2.std():.3f}"
        )
        print(
            f"Permutation Scores rec: {permutation_scores3.mean():.3f} +/- "
            f"{permutation_scores3.std():.3f}"
        )

        os.chdir(dir)
        plt.hist(permutation_scores)
        plt.axvline(score, color='k', linestyle='dashed', linewidth=1)
        plt.title("hist of permutations precision (knn)")
        plt.xlabel("permutation score")
        plt.ylabel("counts")
        plt.savefig("knn_prec_hist.png")
        plt.clf()
        plt.hist(permutation_scores2)
        plt.axvline(score2, color='k', linestyle='dashed', linewidth=1)
        plt.title("hist of permutations recall (knn)")
        plt.xlabel("permutation score")
        plt.ylabel("counts")
        plt.savefig("knn_rec_hist.png")
        plt.clf()
        plt.hist(permutation_scores3)
        plt.axvline(score3, color='k', linestyle='dashed', linewidth=1)
        plt.title("hist of permutations f1 score (knn)")
        plt.xlabel("permutation score")
        plt.ylabel("counts")
        plt.savefig("knn_f1_hist.png")
        plt.clf()