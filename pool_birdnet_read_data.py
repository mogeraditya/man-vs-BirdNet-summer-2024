import os
import glob
import numpy as np
import pandas as pd

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from multiprocessing import Pool
import splicer as sp
import run_birdnet as rb
import split_datasheet as sd
import analysis_of_data as aod
import pickle 

import multiprocessing
import time
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

def make_dir(new_dir): 
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

#given array of names find all unique names
def find_unique_bird_ids(array_w_dicts):
    length= len(array_w_dicts)
    all_common_array=[]
    # all_scientific_array=[]
    for it in range(length):
        current_dict= array_w_dicts[it]
        all_common_array.append(current_dict["common_name"])
    unique= list(set(all_common_array))
    return unique

def store_relevant_info_from_birdnet(array_w_dicts, dir_to_store, audio_file, location, weather, date, conf):
    wd= os.getcwd()

    unique_birds= find_unique_bird_ids(array_w_dicts)
    
    dict_w_start_time= {}
    for bird in unique_birds:
        array_start_time= []
        for it in range(len(array_w_dicts)):
            current_dict= array_w_dicts[it]
            if bird==current_dict["common_name"]:
                array_start_time.append(current_dict["start_time"])
        clean_name= remove_special_from_names([bird])
        dict_w_start_time[clean_name[0]]= array_start_time

    make_dir(dir_to_store)
    os.chdir(dir_to_store)
    new_dir= dir_to_store+ date +"_"+str(conf)+"\\"
    make_dir(new_dir)
    os.chdir(new_dir)
    print(dict_w_start_time)
    with open(audio_file[:-4]+'.pickle', 'wb') as handle:
        pickle.dump(dict_w_start_time, handle, protocol=-1)

    os.chdir(wd)
        
#remove all special characters from name to make common names comparable
def remove_special_from_names(array):
    new_array=[]
    for it in range(len(array)):
        special_string= array[it]
        sample_list=[]
        for i in special_string:
            if i.isalnum()or i==" " or i=="-":
                if i=="-":
                    sample_list.append(" ")
                else:
                    sample_list.append(i.lower())
        # Join the elements in the list to create a string
        normal_string="".join(sample_list)
        new_array.append(normal_string)

    return new_array

def run_birdnet(audio_file, conf_threshold_for_bn, dir,common_resources, location, weather, date, per_date_birdnet_only):
    wd= os.getcwd()
    os.chdir(dir)
    dir= common_resources+ "store_birdnet_all_confs_library\\"
    os.chdir(dir)
    file_name= audio_file[:-4]+'_0_conf.pickle'
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    array= []
    for dict in data:
        if dict["confidence"]>=conf_threshold_for_bn:
            array.append(dict)
    array= np.array(array)
    dir_to_store= common_resources+ "store_birdnet_dictionaries\\"
    store_relevant_info_from_birdnet(array, dir_to_store, audio_file, location, weather, date, conf=conf_threshold_for_bn)

    seg_unique= find_unique_bird_ids(array)
    seg_unique= remove_special_from_names(seg_unique)

    os.chdir(common_resources)
    prediction_df=pd.read_csv("prediction.csv")
    os.chdir(wd)
    prediction_df["common name"]= seg_unique
    prediction_df["Location"]= location
    prediction_df["Weather"]= weather
    file_label= audio_file[-6:-4]
    if file_label=="_0":
        file_label="00"
    if file_label=="_5":
        file_label="05"
    prediction_df["time"]=file_label

    os.chdir(per_date_birdnet_only)
    prediction_df.to_csv(audio_file[:-4]+"_birdnet_5_mins.csv")

    return seg_unique

def get_inputs_to_pool_birdnet(common_resources, split, conf_threshold_for_bn, per_date_birdnet_only):
    master_array= []
    os.chdir(split)
    list_of_folders= glob.glob("**") #indicative of number of dates you are running this on
    number_of_folders= len(list_of_folders)


    for it in range(number_of_folders): #iterate through all the  dates
        folder_name_w_date= list_of_folders[it]
        date= folder_name_w_date[0:8]
        date_in_datetime= datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))

        os.chdir(common_resources)
        date_loc_df= pd.read_csv("date_loc_df.csv")

        date_loc_dict= {str(date_loc_df["date"][i]): [date_loc_df["Location"][i], date_loc_df["Weather"][i], date_loc_df["lat"][i],date_loc_df["lon"][i]] for i in range(len(date_loc_df["date"]))}

        try:    
            location,weather,lat,lon= [date_loc_dict.get(date)[i] for i in (0,1,2,3)]
        except TypeError:
            [location, weather, lat, lon]= ["unknown","unknown", 13.022551345783762, 77.57019815297808]

        date_folder= split+folder_name_w_date+"\\"
        os.chdir(date_folder) #goes to the folder of a particular date

        files_in_a_date= glob.glob("*.WAV")

        file_array= [[file_in_a_date, conf_threshold_for_bn,
                       date_folder, common_resources, location, weather, date, per_date_birdnet_only] for file_in_a_date in files_in_a_date]
        
        master_array+=file_array.copy()
    
    return master_array

def apply_multiprocessing(input_list, input_function, pool_size = 4):
    
    pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=10)

    try:
        jobs = {}
        for value in input_list:
            jobs[value[0]] = pool.apply_async(input_function, value)

        results = {}
        for value, result in jobs.items():
            try:
                results[value] = result.get()
            except KeyboardInterrupt:
                print ("Interrupted by user")
                pool.terminate()
                break
            except Exception as e:
                results[value] = e
        return results
    except Exception:
        raise
    finally:
        pool.close()
        pool.join()

def merged_csv_and_delete_audio(split, per_date_birdnet_only, delete_files):
    df_list= []
    os.chdir(per_date_birdnet_only)
    list_of_csvs= glob.glob("*.csv")
    for csv in list_of_csvs:
        df= pd.read_csv(csv)
        df_list.append(df)
    dataframe= pd.concat(df_list, ignore_index= True)
    dataframe.to_csv("merged_mass_birdnet_5mins.csv")

    if delete_files== True:
        dir= split
        os.chdir(dir)
        for foldername in os.listdir():
            os.chdir(dir+foldername)
            for filename in os.listdir():
                if filename.endswith('.WAV'):
                    os.unlink(filename)

storage= "d:\\Research\\analyze_birdnet\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage#+ "common_resources\\" #refer to common resources folder in github

split= storage+ "audio_files_split\\"
conf= 0.3
per_date_birdnet_only= "D:\\Research\\analyze_birdnet\\folder_for_5mins_0.3confidence\\code\\test_per_date_birdnet_only\\"
file_array= get_inputs_to_pool_birdnet(common_resources, split, conf, per_date_birdnet_only)
if __name__ == "__main__":
    t0 = datetime.now()
    results1 = apply_multiprocessing(file_array, run_birdnet)
    t1 = datetime.now()
    print (results1)
    print ("Time taken for task : {}".format(t1 - t0))