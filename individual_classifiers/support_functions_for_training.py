import os
import glob
import pickle 
import numpy as np
import random
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
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


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

def compare_array(array1, array2):
    tp=0 
    fp=0
    fn=0
    tn=0
    for i in range(len(array1)):
        if array1[i]==1 and array2[i]==1:
            tp+=1
        elif array1[i]==0 and array2[i]==1:
            fp+=1
        elif array1[i]==1 and array2[i]==0:
            fn+=1
        else:
            tn+=1
    return tp,fp,fn,tn

def score_list_of_arrays(list1, list2):
    tp_total, fp_total, fn_total, tn_total= 0,0,0,0
    for i in range(len(list1)):
        tp, fp,fn,tn = compare_array(list1[i],list2[i])
        tp_total+=tp
        fp_total+=fp
        fn_total+=fn
        tn_total+=tn
    recall= tp_total/ (tp_total+fn_total)
    if (tp_total+fp_total) == 0:
        precision=0
        print("ERROR DENOM ZERO")
    else:
        precision= tp_total/(tp_total+fp_total)
        if (precision+recall)==0:
            f1_score= 0
            print("ERROR F1 DENOM ZERO")
        else:
            f1_score= (2*precision*recall)/ (precision+recall)
    return precision, recall, f1_score #'''tp, fp, fn'''

# storage= "d:\\Research\\analyze_embeddings\\" #source folder
# sound_data= storage+ "sound_data\\" #location of sound files

# common_resources= storage+ "\\common_resources\\"
# dir= common_resources+ "store_embeddings_bn_dict\\"
def scorer(clf, x, y):
    y_pred= clf.predict(x)
    tp_total, fp_total, fn_total, tn_total= 0,0,0,0
    for i in range(len(y)):
        tp, fp,fn,tn = compare_array(y[i],y_pred[i])
        tp_total+=tp
        fp_total+=fp
        fn_total+=fn
        tn_total+=tn
    recall= tp_total/ (tp_total+fn_total)
    return recall

def get_x_and_y(dir_embeddings, dir_dataset):

    os.chdir(dir_embeddings)
    list_of_dicts= glob.glob("*.pickle")
    array_of_dicts=[]
    for dicts in list_of_dicts:
        with open(dicts, 'rb') as f:
            dictionary = pickle.load(f)
        array_of_dicts.append(dictionary)
    merged_dictionary= merge_list_of_dicts(array_of_dicts)

    os.chdir(dir_dataset)
    print(dir_dataset)
    with open("y_arr.pickle", 'rb') as f:
        y_dictionary = pickle.load(f)
    with open("loclabels.pickle", 'rb') as f:
        labels_dictionary = pickle.load(f)

    subset_merged_dict= {i:merged_dictionary[i] for i in list(y_dictionary.keys()) if i in merged_dictionary}
    subset_y_dict= {i: y_dictionary[i] for i in list(subset_merged_dict.keys()) if i in y_dictionary}
    subset_labels_dict= {i:labels_dictionary[i] for i in list(subset_merged_dict.keys()) if i in labels_dictionary}
    x= list(subset_merged_dict.values())
    x= [list(i) for i in x]
    y= list(subset_y_dict.values())
    y= [list(i) for i in y]
    groups= list(subset_labels_dict.values())
    groups= [i.strip() for i in groups]
    
    date_groups= [i[0:8] for i in list(subset_merged_dict.keys())]

    return x,y, groups, date_groups
#make sure this is fine ngl
bird_label_list= ['asian brown flycatcher', 'black headed cuckoo shrike', 'blue throated flycatcher', 'common hawk cuckoo', 'lesser cormorant', 'olive backed pipit', 'pond heron', 'spotted owlet', 'sulfur bellied warbler', 'tytlers leaf warbler', 'white rumped shama', 'bayback shrike', 'brahminy kite', 'brown breasted flycatcher', 'house sparrow', 'little swift', 'slaty breasted crake', 'western crowned warbler', 'black headed cuckooshrike', 'green warbler', 'orange headed thrush', 'sykes warbler', 'forest wagtail', 'grey wagtail', 'indian peafowl', 'large billed leaf warbler', 'mottled wood owl', 'black naped oriole', 'greenish warbler', 'house crow', 'tickells leaf warbler', 'asian koel', 'yellow billed babbler', 'white throated kingfisher', 'indian pond heron', 'golden oriole', 'rufous treepie', 'small minivet', 'spot breasted fantail', 'indian blue robin', 'tickells blue flycatcher', 'blue capped rock thrush', 'indian pitta', 'shikra', 'booted warbler', 'jungle myna', 'black naped monarch', 'white browed bulbul', 'black drongo', 'indian paradise flycatcher', 'common tailorbird', 'indian white eye', 'chestnut tailed starling', 'cinerous tit', 'pale billed flowerpecker', 'coppersmith barbet', 'verditer flycatcher', 'rock pigeon', 'scaly breasted munia', 'greater coucal', 'blythes reed warbler', 'purple sunbird', 'lottens sunbird', 'ashy drongo', 'ashy prinia', 'rose ringed parakeet', 'common myna', 'spotted dove', 'booted eagle', 'oriental magpie robin', 'purple rumped sunbird', 'white cheeked barbet', 'large billed crow', 'red whiskered bulbul', 'black kite']

def get_xy_for_individual_classsifier(dir_embeddings, dir_dataset, species_name):
    x,y, groups, date_groups= get_x_and_y(dir_embeddings, dir_dataset)
    #look at each data point in y; if  bird in y[i] then pull x[i], label as positive; else label as negative
    index_of_bird= bird_label_list.index(species_name)
    pos_neg_labels= []
    positive_x, negative_x= [], []
    pos_group, neg_group= [], []
    pos_date, neg_date= [], []
    for i in range(len(y)):
        if y[i][index_of_bird]==1:
            pos_neg_labels.append("positive")
            positive_x.append(x[i])
            pos_group.append(groups[i])
            pos_date.append(date_groups[i])
        else:
            pos_neg_labels.append("negative")
            negative_x.append(x[i])
            neg_group.append(groups[i])
            neg_date.append(date_groups[i])
    print(len(y))    
    # print(dict_labels)
    number_of_positives= len(positive_x)
    print("number of positives for "+species_name+" = " +str(number_of_positives))
    #we sample as many negatives as positives
    sampled_negs_index= random.sample(range(len(negative_x)), number_of_positives)
    sampled_negs= [negative_x[i] for i in sampled_negs_index]
    sampled_negs_groups= [neg_group[i] for i in sampled_negs_index]
    sampled_negs_date= [neg_date[i] for i in sampled_negs_index]
    
    pos_score= list(np.ones(shape=(len(positive_x))))
    neg_score= list(np.zeros(shape=(len(sampled_negs))))
    
    Score= pos_score + neg_score
    X= positive_x + sampled_negs
    groups= list(pos_group) + list(sampled_negs_groups)
    date_groups= list(pos_date) + list(sampled_negs_date)
    print(len(groups), len(positive_x))
    return X, Score, groups, date_groups

def evaluate_model(data_x, data_y):
    k_fold = KFold(3, shuffle=True, random_state=1)
    predicted_targets = np.array([])
    actual_targets = np.array([])
    for train_ix, test_ix in k_fold.split(data_x):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]

        classifier = KNeighborsClassifier(n_jobs=4, n_neighbors=12)
        predicted_labels = classifier.predict(test_x)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets

def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix

def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
def get_corr_incorr_counts(date_groups, unique_dates, y_pred, y_test):
    
    count_list_correct= np.zeros(shape= (len(unique_dates))); count_list_incorrect= np.zeros(shape= (len(unique_dates)))
    for it in range(len(y_pred)):
        get_date= date_groups[it]
        get_index= unique_dates.index(get_date)
        add_list= np.zeros(shape= (len(unique_dates)))
        add_list[get_index]=1
        if y_pred[it]==y_test[it]:
            count_list_correct+=add_list
        else:
            count_list_incorrect+=add_list
    return count_list_correct, count_list_incorrect