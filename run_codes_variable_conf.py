import os
import glob
import numpy as np
import pandas as pd

#importing python files
import splicer as sp
import run_birdnet as rb
import split_datasheet as sd
import analysis_of_data as aod


storage= "d:\\Research\\Rohini 2024\\Bird Data\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\" #refer to common resources folder in github
interval_of_conf= np.arange(0.4, 0.9, 0.1) #input the ranges of conf to run birdnet over

time_interval= 5 #in minutes

for it in range(len(interval_of_conf)):
    conf_threshold_for_bn= interval_of_conf[it] #ranges from 0 to 1

    new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf_threshold_for_bn)

    os.chdir(sound_data)
    sound_files_names= glob.glob("*.WAV")

    os.chdir(common_resources)
    df= pd.read_excel("current_training.xlsx")

    sp.run_slicer_on_all_files(df, sound_data, time_interval, split)
    rb.mass_run_birdnet(split, common_resources, time_interval, conf_threshold_for_bn, per_date_birdnet_only, delete_files=True)

    os.chdir(common_resources)
    df_datasheet= pd.read_csv("current_training_w_codes.csv")
    df_datasheet= sd.add_date_time_codes(df_datasheet)
    sd.parse_datasheet_alldates_into_files(df_datasheet, time_interval, per_date_birdnet_only, datasheet_per_date_birdnet_only)

    dir_store_confusion= common_resources+ "conf_" + str(interval_of_conf[0])+ "to" + str(interval_of_conf[-1]) + "\\"
    aod.generate_confusion_matrix_info(time_interval, conf_threshold_for_bn, datasheet_per_date_birdnet_only, per_date_birdnet_only, dir_store_confusion)