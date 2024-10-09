import os
import glob
import pickle 
import numpy as np
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
    
    date_groups= [i[0:8] for i in list(subset_merged_dict.keys())]

    return x,y, groups


