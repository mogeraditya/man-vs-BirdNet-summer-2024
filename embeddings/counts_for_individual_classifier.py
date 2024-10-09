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

list_of_birds= ['asian brown flycatcher', 'black headed cuckoo shrike', 'blue throated flycatcher', 'common hawk cuckoo', 'lesser cormorant', 'olive backed pipit', 'pond heron', 'spotted owlet', 'sulfur bellied warbler', 'tytlers leaf warbler', 'white rumped shama', 'bayback shrike', 'brahminy kite', 'brown breasted flycatcher', 'house sparrow', 'little swift', 'slaty breasted crake', 'western crowned warbler', 'black headed cuckooshrike', 'green warbler', 'orange headed thrush', 'sykes warbler', 'forest wagtail', 'grey wagtail', 'indian peafowl', 'large billed leaf warbler', 'mottled wood owl', 'black naped oriole', 'greenish warbler', 'house crow', 'tickells leaf warbler', 'asian koel', 'yellow billed babbler', 'white throated kingfisher', 'indian pond heron', 'golden oriole', 'rufous treepie', 'small minivet', 'spot breasted fantail', 'indian blue robin', 'tickells blue flycatcher', 'blue capped rock thrush', 'indian pitta', 'shikra', 'booted warbler', 'jungle myna', 'black naped monarch', 'white browed bulbul', 'black drongo', 'indian paradise flycatcher', 'common tailorbird', 'indian white eye', 'chestnut tailed starling', 'cinerous tit', 'pale billed flowerpecker', 'coppersmith barbet', 'verditer flycatcher', 'rock pigeon', 'scaly breasted munia', 'greater coucal', 'blythes reed warbler', 'purple sunbird', 'lottens sunbird', 'ashy drongo', 'ashy prinia', 'rose ringed parakeet', 'common myna', 'spotted dove', 'booted eagle', 'oriental magpie robin', 'purple rumped sunbird', 'white cheeked barbet', 'large billed crow', 'red whiskered bulbul', 'black kite']

storage= "d:\\Research\\analyze_embeddings\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\"
plot= storage+ "plots\\grouped_by_days\\"
dir_heard= common_resources+ "heard_datafiles\\"
dir_seen= common_resources+ "seen_datafiles\\"

dir_bn= common_resources+ "store_embeddings_bn_dict\\"
dir_bn_stack= common_resources+ "store_embeddings_bn_stack_dict\\"
dir_vggish= common_resources+ "store_vggish_embeddings\\"
dir_vggish_stack= common_resources+ "store_vggish_embeddings_stacked\\"

loop1, labels1= [dir_seen], ["seen"]
loop2, labels2= [dir_bn], ["bn"]

for item1 in range(len(loop1)):
    for item2 in range(len(loop2)):
        dir= plot+labels1[item1]+"_"+labels2[item2]+"\\"
        big_label= labels1[item1]+"_"+labels2[item2]
        make_dir(dir)
        os.chdir(dir)
        x,y,groups= sf.get_x_and_y(loop2[item2], loop1[item1])
        sum_list = np.zeros(shape=(len(y[0])))
        for i in range(len(y)):
            sum_list+= np.array(y[i])
        print(sum_list)
        # plt.hist(sum_list)
        # for i in range(len(sum_list)):
        #     if sum_list[i]>=90:
        #         print("bird name: "+str(list_of_birds[i])+ ", count: " +str(sum_list[i]))
        # sub_sum_list= [sum_list[i] for i in range(len(sum_list)) if sum_list[i]>=90]
        # sub_bird_list= [list_of_birds[i] for i in range(len(sum_list)) if sum_list[i]>=90]
        plt.bar(np.arange(len(sum_list)), sum_list)
        # plt.xticks(rotation=45, ha='right')
        plt.plot(np.ones(shape=(len(sum_list)))*90, color='red', linestyle='dashed')
        plt.tight_layout()
        plt.show()
        plt.clf()