import os
import glob
import numpy as np
import pandas as pd

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import pickle 
import multiprocessing
from datetime import datetime
# import splicer as sp
# import generate_confusion_data as gcd
# import fetch_files_support_functions as ff



analyzer = Analyzer()

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

def create_embeddings_averaged_one_minute(array_of_dicts, dir_to_store, date_code):
    embedding_arr= []
    keys= []
    for time in range(0,60):
        array_sub_time= list(np.arange(time*20, (time+1)*20))
        null_vector= np.zeros(shape=(1024))
        counter=0
        for sub_time in array_sub_time:
            null_vector= np.add(null_vector, array_of_dicts[sub_time]["embeddings"])
            counter+=1
        average_embedding_vector= null_vector/counter
        embedding_arr.append(average_embedding_vector)
        print(len(average_embedding_vector))
        keys.append(time)
    dictionary= dict(zip(keys, embedding_arr))
    # print(dictionary)

    make_dir(dir_to_store)
    os.chdir(dir_to_store)
    with open(str(date_code)+ "_embeddings.pickle", 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=-1)
    
    return dictionary

def generate_embeddings_per_date(audio_file,lat, lon, date_code, date_in_datetime, conf, common_resources, sound_data):
    recording = Recording(
    analyzer,
    sound_data+audio_file,
    lat= lat,
    lon= lon,
    date= date_in_datetime, # use date or week_48
    min_conf= conf,
    )
    recording.extract_embeddings()
    embeddings= recording.embeddings

    dir_to_store= common_resources+"store_embeddings_dict\\"+ date_code + "\\"
    create_embeddings_averaged_one_minute(embeddings, dir_to_store, date_code)
    return "Done for " + str(date_code)

def get_inputs_to_pool_birdnet(common_resources, sound_data, conf):
    master_array= []    
    os.chdir(sound_data)
    list_of_dates= glob.glob("*.WAV") #indicative of number of dates you are running this on
    list_of_dates= [i[0:8] for i in list_of_dates]
    for date in list_of_dates: #iterate through all the  dates
        date_in_datetime= datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))
        os.chdir(common_resources)
        date_loc_df= pd.read_csv("date_loc_df.csv")
        date_loc_dict= {str(date_loc_df["date"][i]): [date_loc_df["Location"][i], date_loc_df["Weather"][i], date_loc_df["lat"][i],date_loc_df["lon"][i]] for i in range(len(date_loc_df["date"]))}
        try:    
            location,weather,lat,lon= [date_loc_dict.get(date)[i] for i in (0,1,2,3)]
        except TypeError:
            [location, weather, lat, lon]= ["unknown","unknown", 13.022551345783762, 77.57019815297808]
        os.chdir(sound_data)
        file_name= glob.glob(str(date)+"*")[0]
        file_array= [file_name,lat, lon, date, date_in_datetime, conf, common_resources, sound_data]
        master_array.append(file_array)
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

storage= "d:\\Research\\analyze_embeddings\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
wd= "D:\\github\\man-vs-BirdNet-summer-2024-\\embeddings"
common_resources= wd+ "\\common_resources\\"
dir= common_resources+ "store_embedding_dictionaries\\"
conf=0.25
file_array= get_inputs_to_pool_birdnet(common_resources, sound_data, conf)
print(file_array)
if __name__ == "__main__":
    t0 = datetime.now()
    results1 = apply_multiprocessing(file_array, generate_embeddings_per_date)
    t1 = datetime.now()
    print (results1)
    print ("Time taken for task : {}".format(t1 - t0))