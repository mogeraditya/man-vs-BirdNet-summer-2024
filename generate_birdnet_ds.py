import os
import glob
import numpy as np
import pandas as pd

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from multiprocessing import Pool
import splicer as sp
from splicer import make_dir
import run_birdnet as rb
import split_datasheet as sd
import analysis_of_data as aod
import pickle 

import multiprocessing
import time
from datetime import datetime

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

conf=0.0

def run_birdnet(audio_file, lat, lon, date_in_datetime, conf_threshold_for_bn, dir, common_resources):
    wd= os.getcwd()
    os.chdir(dir)
    recording = Recording(
    analyzer,
    audio_file,
    lat= lat,
    lon= lon,
    date=date_in_datetime, # use date or week_48
    min_conf= conf_threshold_for_bn,
    )
    recording.analyze()

    dict= recording.detections
    dir_to_store= common_resources+ "store_birdnet_all_confs_library\\"
    make_dir(dir_to_store)
    os.chdir(dir_to_store)
    with open(audio_file[:-4]+'_0_conf.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=-1)


def get_inputs_to_pool_birdnet(common_resources, split, conf_threshold_for_bn):
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

        file_array= [[file_in_a_date,lat, lon, date_in_datetime, conf_threshold_for_bn,
                       date_folder, common_resources] for file_in_a_date in files_in_a_date]
        
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

storage= "d:\\Research\\analyze_birdnet\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\" #refer to common resources folder in github

split= storage+ "audio_files_split\\"

file_array= get_inputs_to_pool_birdnet(common_resources, split, conf)
if __name__ == "__main__":
    t0 = datetime.now()
    results1 = apply_multiprocessing(file_array, run_birdnet)
    t1 = datetime.now()
    print (results1)
    print ("Time taken for task : {}".format(t1 - t0))