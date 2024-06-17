import os
import glob
import pandas as pd
import datetime
#importing python files
import splicer as sp
import pool_birdnet as pb
import split_datasheet as sd
import analysis_of_data as aod

storage= "d:\\Research\\Rohini 2024\\Bird Data\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\" #refer to common resources folder in github

time_interval= 5 #in minutes
conf_threshold_for_bn= 0.3 #ranges from 0 to 1

new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf_threshold_for_bn)

# os.chdir(sound_data)
# sound_files_names= glob.glob("*.WAV")

os.chdir(common_resources)
df= pd.read_excel("current_training.xlsx")
split= storage+ "audio_files_split\\"
# sp.run_slicer_on_all_files(df, sound_data, time_interval, split)

os.chdir(split)
list_of_folders= glob.glob("**") #indicative of number of dates you are running this on
number_of_folders= len(list_of_folders)
print(list_of_folders)

for it in range(number_of_folders): #iterate through all the  dates
    folder_name_w_date= list_of_folders[it]
    file_array= pb.get_inputs_to_pool_birdnet(folder_name_w_date, common_resources, split, conf_threshold_for_bn)
    if __name__ == "__main__":
        t0 = datetime.now()
        results1 = pb.apply_multiprocessing(file_array, pb.run_birdnet)
        t1 = datetime.now()
        print (results1)
        print ("Multi: {}".format(t1 - t0))