import os
import glob
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import pickle
import splicer as sp
import generate_confusion_data as gcd
import fetch_files_support_functions as ff

storage= "d:\\Research\\analyze_birdnet\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage#+ "common_resources\\" #refer to common resources folder in github
dir= common_resources+ "store_birdnet_dictionaries\\"

time_interval=5 
# conf=0.3
interval_of_conf= list(np.arange(0.3, 0.9, 0.02)) #input the ranges of conf to run birdnet over
interval_of_conf= [np.round(i,2) for i in interval_of_conf]

for conf in interval_of_conf:
    new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf)
    split= storage+ "audio_files_split\\"
    confusion_codes= ["tp", "fp", "fn"]
    datasheet_per_date_birdnet_only= "D:\\Research\\analyze_birdnet\\datasheet_per_date_birdnet_only\\"
    for confusion_code in confusion_codes:
        gcd.mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf, confusion_code)
    
# confusion_type= "fp"
# ff.mass_run_fetch_files(confusion_type, conf, code, split, dir)

