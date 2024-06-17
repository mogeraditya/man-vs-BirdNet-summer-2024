import os
import glob
import numpy as np
import pandas as pd

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
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

#run birdnet on all the audio file segments (for each day for each time interval audio piece)
def mass_run_birdnet(split, common_resources, time_interval, conf_threshold_for_bn, per_date_birdnet_only, delete_files):
    #initialize dataframes
    os.chdir(common_resources)
    prediction_df=pd.read_csv("prediction.csv")
    superer_df=pd.read_csv("prediction.csv")

    os.chdir(split)
    list_of_folders= glob.glob("**") #indicative of number of dates you are running this on
    number_of_folder= len(list_of_folders)

    for ti in range(number_of_folder):
        date= list_of_folders[ti]
        date_in_datetime= datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))

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
                
        os.chdir(common_resources)
        prediction_df= pd.read_csv("prediction.csv")
        super_df= pd.read_csv("prediction.csv")

        date_folder= split+list_of_folders[ti]+"\\"
        os.chdir(date_folder) #goes to the folder of a particular date

        files_in_a_date= glob.glob("*.WAV")
        number_of_files_per_date=len(files_in_a_date)
        
        for it in range(number_of_files_per_date):
            
            recording = Recording(
            analyzer,
            files_in_a_date[it],
            lat= lat,
            lon= lon,
            date=date_in_datetime, # use date or week_48
            min_conf= conf_threshold_for_bn,
            )
            recording.analyze()

            array= np.array(recording.detections)

            seg_unique= find_unique_bird_ids(array)
            # number_of_unique_birds= len(seg_unique)

            wd= os.getcwd()
            os.chdir(common_resources)
            prediction_df=pd.read_csv("prediction.csv")
            os.chdir(wd)

            prediction_df["common name"]= remove_special_from_names(seg_unique)
            prediction_df["Location"]= location
            prediction_df["Weather"]= weather
            file_label= files_in_a_date[it][-6:-4]
            if file_label=="_0":
                file_label="00"
            if file_label=="_5":
                file_label="05"
            prediction_df["time"]=file_label

            super_df= pd.concat([super_df,prediction_df])
        
        super_df["date"]=list_of_folders[ti][0:8]
        superer_df = pd.concat([superer_df,super_df])
        wd= os.getcwd()
        
        os.chdir(per_date_birdnet_only)
        super_df.to_csv(list_of_folders[ti][0:8]+"_split_"+str(time_interval)+"mins.csv")
        os.chdir(wd)

    os.chdir(per_date_birdnet_only)
    superer_df.to_csv("merged_mass_birdnet_"+str(time_interval)+"mins.csv")

    if delete_files== True:
        dir= split
        os.chdir(dir)
        for foldername in os.listdir():
            os.chdir(dir+foldername)
            for filename in os.listdir():
                if filename.endswith('.WAV'):
                    os.unlink(filename)

