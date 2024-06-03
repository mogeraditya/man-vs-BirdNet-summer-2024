import soundfile as sf
import librosa
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import random
import matplotlib as mpl

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

import shutil
import run_birdnet as rb

# Load and initialize the BirdNET-Analyzer models.
# analyzer = Analyzer()


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

def mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf):
    #given
    os.chdir(datasheet_per_date_birdnet_only)
    file_list= glob.glob("*csv")
    dates= [i[0:8] for i in file_list]
    dates=dates[:-1]

    super_stats_df=pd.DataFrame(columns=["date", "# birdnet\\datasheet", "time code" ])

    for iterant in range(len(dates)):
        date= dates[iterant]

        os.chdir(datasheet_per_date_birdnet_only)
        datasheet_split = glob.glob(date+"*")
        print(datasheet_split)
        df_ds_split= pd.read_csv(datasheet_split[0])

        os.chdir(per_date_birdnet_only)
        birdnet_split = glob.glob(date+"*")
        df_bd_split= pd.read_csv(birdnet_split[0])

        array_times= df_ds_split['time code']
        array_times=[str(i) for i in array_times]
        array_times=["0"+i if len(i)<2 else i for i in array_times]
        df_ds_split['time code']= array_times
        df_ds_split
        array_times= df_bd_split['time']
        array_times=[str(i) for i in array_times]
        array_times=["0"+i[1] if i[0]=="_" else i for i in array_times]
        df_bd_split['time']= array_times

        df_bd_split["common name"]=rb.remove_special_from_names(df_bd_split["common name"])
        df_bd_split

        date= dates[iterant]

        stats_df= pd.DataFrame(columns=["date","# birdnet\\datasheet", "time code"])

        input_time_interval=5

        time_codes= np.arange(0, 60, input_time_interval)

        time_codes=[str(i) for i in time_codes]
        time_codes=["0"+i if len(i)<2 else i for i in time_codes]

        # bd_minus_ds_array=[]

        print(date)
        for it in range(len(time_codes)):
            time= time_codes[it]
            new_df= pd.DataFrame(columns=["date","# birdnet\\datasheet", "time code"])
            ds_groupby= df_ds_split.groupby("time code")
            ds_compare= ds_groupby.get_group(time_codes[it])
            ds_compare_array=set(np.array(ds_compare["common name"]))
            

            bd_groupby= df_bd_split.groupby("time")
            try:
                bd_compare= bd_groupby.get_group(time_codes[it])
                bd_compare_array=set(np.array(bd_compare["common name"]))
            except KeyError:
                bd_compare_array=set([])

            bd_minus_ds= bd_compare_array.copy()
            bd_minus_ds= bd_minus_ds.difference(ds_compare_array)
            new_df["# birdnet\\datasheet"]= list(bd_minus_ds)
            new_df["date"]= date
            new_df["time code"]=time
            stats_df= pd.concat([stats_df, new_df])
        stats_df["date"]= date
        # stats_df.to_csv(date+"_stats_w_names_cl.csv")  
        super_stats_df= pd.concat([super_stats_df, stats_df])

    dir= code+"\\stats_fp\\"
    make_dir(dir)
    os.chdir(dir)
    super_stats_df.to_csv("all_dates_merged_w_names_"+str(conf)+"confidence.csv")
    return "done :D"

def copy_required_files(code, split, conf, delete_files):
    os.chdir(code+"\\stats_fp\\")
    df_w_differences= pd.read_csv("all_dates_merged_w_names_"+str(conf)+"confidence.csv")
    range_of_times= df_w_differences["time code"]
    range_of_times=[str(i) for i in range_of_times]
    range_of_times=["0"+i if len(i)<2 else i for i in range_of_times ]
    df_w_differences["time code"]= range_of_times
    array_unique= df_w_differences["# birdnet\\datasheet"]
    array_unique= list(set(array_unique))
    array_unique
    array_data= df_w_differences["# birdnet\\datasheet"]

    samples_audio_bnfails= code+ "\\audio_files_fp\\"
    make_dir(samples_audio_bnfails)

    os.chdir(split)
    array_w_days= glob.glob("*")
    array_w_cut_days= [i[0:8] for i in array_w_days]
    for it in range(len(array_unique)):
        for jt in range(len(array_data)):
            if array_data[jt]==array_unique[it]:
                bird_name= array_unique[it]

                bird_dir= samples_audio_bnfails +"\\"+ bird_name #+"\\"
                if not os.path.exists(bird_dir):
                    os.makedirs(bird_dir)

                date= str(df_w_differences["date"][jt])
                time_code= str(df_w_differences["time code"][jt])
                filename= str(date) + "_split_"+ str(time_code)+".WAV"

                for kt in range(len(array_w_cut_days)):
                    if array_w_cut_days[kt]==date:
                        dir_to_copy= split+array_w_days[kt]
                        new_dir= bird_dir +"\\"+ date #+"\\"
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                        old_path= dir_to_copy+"\\"+filename
                        new_path= new_dir+"\\"+filename
                        shutil.copy(old_path, new_path)
                        print('Copied')

    
    if delete_files== True:
        dir= split
        os.chdir(dir)
        for foldername in os.listdir():
            os.chdir(dir+foldername)
            for filename in os.listdir():
                if filename.endswith('.WAV'):
                    os.unlink(filename)

    print("all done :D")
    return "done"
