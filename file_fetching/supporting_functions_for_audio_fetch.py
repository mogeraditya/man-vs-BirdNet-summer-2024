
import os
import glob
import pandas as pd

#importing python files
import splicer as sp
import run_birdnet as rb
import split_datasheet as sd
import analysis_of_data as aod

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
from splicer import make_dir
import run_birdnet as rb
import generate_false_positive_audio as gfp
import generate_true_positive_audio as gtp
import generate_false_negative_audio as gfn


analyzer= Analyzer()


# storage= "d:\\Research\\Rohini 2024\\Bird Data\\" #source folder
# sound_data= storage+ "sound_data\\" #location of sound files
# common_resources= storage+ "common_resources\\" #refer to common resources folder in github

# time_interval= 5 #in minutes
# conf_threshold_for_bn= 0.9 #ranges from 0 to 1

# new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf_threshold_for_bn)

# confusion = ["tp","fn","fp"]

def run_required_code(confusion_entry, code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf_threshold_for_bn, split, delete_files):
    samples_audio= code+ "audio_files_"+str(confusion_entry)+"\\"
    big_dir= code+"animals_"+str(confusion_entry)+"_files\\"
    make_dir(big_dir)
    if confusion_entry=="fp":
        gfp.mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf_threshold_for_bn)
        gfp.copy_required_files(code, split, conf_threshold_for_bn, delete_files)
    if confusion_entry=="fn":
        gfn.mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf_threshold_for_bn)
        gfn.copy_required_files(code, split, conf_threshold_for_bn, delete_files)
    if confusion_entry=="tp":
        gtp.mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf_threshold_for_bn)
        gtp.copy_required_files(code, split, conf_threshold_for_bn, delete_files)

    return samples_audio, big_dir

def get_piece_of_audio(date_folder, audio_file, start_time, end_time, bird, store_fp_pieces_dir, kv):
    # os.chdir(sound_folder)
    os.chdir(date_folder)
    audio, sr = librosa.load(audio_file)

    slice= audio[start_time*sr: end_time*sr]
    out_filename =  audio_file[0:8] + "_piece_of_"+str(bird)+"_"+str(kv)+".WAV"
    os.chdir(store_fp_pieces_dir)
    sf.write(out_filename, slice, sr)
    print("pieced "+ bird)
    return None

def run_birdnet_and_fetch_files(samples_audio, big_dir, common_resources, conf_threshold_for_bn):
    os.chdir(samples_audio)
    list_of_folders= glob.glob("**") #different animals you are running this on
    number_of_folders= len(list_of_folders)


    # lat,lon= 13.022551345783762, 77.57019815297808 #fixing lat lot cause why not
    for iv in range(number_of_folders):
        bird= list_of_folders[iv]
        bird_folder= samples_audio+bird+"\\"
        os.chdir(bird_folder) #goes to the folder of a particular bird
        dates_within_animal= glob.glob("**")
        dates= dates_within_animal
        dates_in_datetime= [datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])) for date in dates]
        number_of_folders_per_bird=len(dates_within_animal)
        # tab
        store_fp_pieces_dir= big_dir+str(bird)+"_fp_files\\"
        make_dir(store_fp_pieces_dir)
        for jv in range(number_of_folders_per_bird):
            date= dates_within_animal[jv]
            date_in_datetime= dates_in_datetime[jv]
            date_folder= bird_folder+date+"\\"
            # print(date)

            os.chdir(common_resources)
            date_loc_df= pd.read_csv("date_loc_df.csv")
            for i in range(len(date_loc_df["date"])):
                
                if str(date)== str(date_loc_df["date"][i]):
                    lat= date_loc_df["lat"][i]
                    lon= date_loc_df["lon"][i]
                    location= date_loc_df["Location"][i]
                    weather= date_loc_df["Weather"][i]
                    break

                else:
                    lat,lon= 13.022551345783762, 77.57019815297808
                    location= "unknown"
                    weather= "unknown"
            print(date_folder)
            os.chdir(date_folder)
            audio_files= glob.glob("*.WAV")
            number_of_audio_files= len(audio_files)
            for kv in range(number_of_audio_files):
                os.chdir(date_folder)
                audio_file= audio_files[0]
                recording = Recording(
                analyzer,
                audio_file,
                lat= lat,
                lon= lon,
                date=date_in_datetime, # use date or week_48
                min_conf= conf_threshold_for_bn,
                )
                recording.analyze()
                array= np.array(recording.detections)
                # rb.remove_special_from_names()
                array_w_dicts= array.copy()
                length= len(array_w_dicts)
                all_common_array=[]
                tag= audio_file[-6:-4]
                # all_scientific_array=[]
                for it in range(length):
                    current_dict= array_w_dicts[it]
                    all_common_array.append(current_dict["common_name"])
                array_w_all_detections= rb.remove_special_from_names(all_common_array)
                for it in range(len(array_w_all_detections)):
                    if array_w_all_detections[it]==bird:
                        start_time= int(array[it]["start_time"])
                        end_time= int(array[it]["end_time"])
                        print(start_time, end_time)
                get_piece_of_audio(date_folder, audio_file, start_time, end_time, bird, store_fp_pieces_dir, tag)