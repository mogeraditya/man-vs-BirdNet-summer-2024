import numpy as np
import pickle
import os
import random
import support_functions_for_training as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import multiprocessing

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
loop2, labels2= [dir_bn_stack, dir_vggish, dir_vggish_stack], ["bn", "bn_stack", "vggish", "vggish_stack"]

inputs= []
for item1 in range(len(loop1)):
    for item2 in range(len(loop2)):
        inputs.append((item1,item2))
        dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
        big_label= labels1[item1]+"_"+labels2[item2]
        make_dir(dir)
        os.chdir(dir)
        x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
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
        for i in range(0,500):
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
        save_array("permutation_precision", permutation_scores)
        save_array("permutation_recall", permutation_scores2)
        save_array("permutation_f1_score", permutation_scores3)
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

# def apply_multiprocessing(input_list, input_function, pool_size = 4):
    
#     pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=10)

#     try:
#         jobs = {}
#         for value in input_list:
#             jobs[value[0]] = pool.apply_async(input_function, value)

#         results = {}
#         for value, result in jobs.items():
#             try:
#                 results[value] = result.get()
#             except KeyboardInterrupt:
#                 print ("Interrupted by user")
#                 pool.terminate()
#                 break
#             except Exception as e:
#                 results[value] = e
#         return results
#     except Exception:
#         raise
#     finally:
#         pool.close()
#         pool.join()

# if __name__ == "__main__":
#     t0 = datetime.now()
#     results1 = apply_multiprocessing(inputs, permute)
#     t1 = datetime.now()
#     print (results1)
#     print ("Time taken for task : {}".format(t1 - t0))