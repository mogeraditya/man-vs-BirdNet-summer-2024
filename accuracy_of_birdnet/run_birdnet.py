import os
import glob
import pandas as pd
from datetime import datetime
import numpy as np
#importing python files
import splicer as sp
import split_datasheet as sd
import generate_confusion_data as gcd
from generate_confusion_data import mass_run_all_dates
import pool_birdnet_read_data as pbrd
import fetch_files_support_functions as ff

storage= "d:\\Research\\analyze_birdnet\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage#+ "common_resources\\" #refer to common resources folder in github

interval_of_conf= [0.3] #list(np.arange(0.3, 0.9, 0.02)) #input the ranges of conf to run birdnet over
interval_of_conf= [np.round(i,2) for i in interval_of_conf]
time_interval= 5 #in minutes
confusion_entries= ["tp", "fn", "fp"]

os.chdir(common_resources)
df_datasheet= pd.read_csv("current_training_dataset.csv")
df_datasheet= sd.add_date_time_codes(df_datasheet)
datasheet_per_date_birdnet_only= "D:\\Research\\analyze_birdnet\\datasheet_per_date_birdnet_only\\"
per_date_birdnet_only= "D:\\Research\\analyze_birdnet\\folder_for_5mins_0.3confidence\\code\\per_date_birdnet_only\\"
df_datasheet=sd.add_required_start_time_codes(df_datasheet)
sd.parse_datasheet_alldates_into_files(df_datasheet, time_interval, per_date_birdnet_only, datasheet_per_date_birdnet_only, common_resources)

# for conf in interval_of_conf:
#     new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, str(np.round(conf,2)))
#     split= storage+ "audio_files_split\\"

# #     # os.chdir(common_resources)
# #     # df_datasheet= pd.read_excel("current_training_dataset.xlsx")
# #     # df_datasheet= sd.add_date_time_codes(df_datasheet)
# #     # sd.parse_datasheet_alldates_into_files(df_datasheet, time_interval, per_date_birdnet_only, datasheet_per_date_birdnet_only)
# #     datasheet_per_date_birdnet_only= "D:\\Research\\analyze_birdnet\\datasheet_per_date_birdnet_only\\"
#     dir_store_confusion= common_resources+ "conf_" + str(conf)+ "\\"
#     for confusion in confusion_entries:
#         mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf, confusion)

#     confusion_type= "fp"
#     ff.mass_run_fetch_files(confusion_type, conf, code, split, dir)