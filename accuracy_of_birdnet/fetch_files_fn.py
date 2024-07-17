import os
import glob
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import pickle
import split_datasheet as sd
import splicer as sp
import random
# import generate_confusion_data as gcd
# import fetch_files_support_functions as ff


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

def add_date_time_codes(df_datasheet):
    array= df_datasheet["Date"]
    # print(array[0])
    array= [int(str(i)[0:4]+str(i)[5:7]+str(i)[8:10]) for i in array]
    time_array=df_datasheet["Time"]
    time_array=[str(i)[3:5] for i in time_array]
    df_datasheet["date code"]= array
    df_datasheet["time code"]= time_array
    return df_datasheet

def get_start_time_info_per_bird_per_timecode(dir, date, time_code, bird):
    time_str= str(time_code)
    if time_str== "0":
        time_str="00"
    if time_str=="5":
        time_str= "05"
    
    os.chdir(dir)
    file_name= glob.glob(str(date)+"_split_"+str(time_str)+".pickle")[0]
    print(file_name)
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    try:
        array_start_time= data[bird]
        print(array_start_time)
    except KeyError:
        array_start_time= []
    return array_start_time

def get_piece_of_audio(in_folder, audio_file, start_time, bird, dir_to_store, split):
    # os.chdir(sound_folder)
    os.chdir(split)
    os.chdir(in_folder)
    audio, sr = librosa.load(audio_file)

    end_time= start_time+60
    slice= audio[int(start_time)*sr: int(end_time)*sr]
    out_filename =  audio_file[:-4] +"_"+str(start_time) +"_piece_of_"+str(bird)+ ".WAV"

    os.chdir(dir_to_store)
    sf.write(out_filename, slice, sr)
    print("pieced "+ bird)
    return out_filename

def fetch_files(dir, date, time_code, bird, dir_tp_files, split):
    array_start_time= get_start_time_info_per_bird_per_timecode(dir, date, time_code, bird)
    #open_data_sheet
    # os.chdir(datasheet_per_date_birdnet_only)
    # df_datasheet= "5mintask_merged_datasheet.csv"
    # date_df= df_datasheet.groupby("date").get_group(date)
    # time_date_df= date_df.groupby("time code").get_group(time_code)

    dir_to_store= dir_tp_files + bird+ "\\"
    make_dir(dir_to_store)
    #find date folder in audio split
    os.chdir(split)

    input_folder_name= glob.glob(str(date)+"*")[0]

    audio_file_name= date+"_split_"+time_code+".WAV"

    if len(array_start_time)==0:
        return [],[]
    else:
        out_filenames=[]
        for start_time in array_start_time:
            out_filename= get_piece_of_audio(input_folder_name, audio_file_name, start_time, bird, dir_to_store, split)
            out_filenames.append(out_filename)
        return array_start_time, out_filenames

def mass_run_fetch_files(confusion_type, conf, code, split, dir):
    dir_tp_files= code + confusion_type+"_audio_files\\"
    make_dir(dir_tp_files)
    stats= code+"\\stats_"+confusion_type+"\\"
    os.chdir(stats)
    merged_dataframe= pd.read_csv("all_dates_merged_w_names_"+str(conf)+"confidence.csv")
    grouped_df= merged_dataframe.groupby("name")
    unique_bird_names= list(set(list(merged_dataframe["name"])))
    # df_store_info_fp= pd.DataFrame(columns=["name", "start time", "audio file name", "present/absent (1/0)"])
    df_list=[]
    for bird in unique_bird_names:
        df_store_info = pd.DataFrame(columns=["name", "start time", "audio file name", "present/absent (1/0)", "date", "time"])
        bird_dir= dir_tp_files+ bird+ "\\"
        make_dir(bird_dir)
        bird_df= grouped_df.get_group(bird)
        date_list, time_list= list(bird_df["date"]), list(bird_df["time code"])
        sup_date_time_tuple_list= [(date_list[i],time_list[i]) for i in range(len(bird_df["name"]))]
        df_list_2= []

        if len(bird_df["date"])>10:
            number_of_samples= 10
        else: 
            number_of_samples= len(bird_df["date"])

        date_time_tuple_list=[]
        
        if len(sup_date_time_tuple_list)<=number_of_samples:
            date_time_tuple_list=sup_date_time_tuple_list
        else: 
            
            indices= random.sample(range(0,len(sup_date_time_tuple_list)), number_of_samples)
            date_time_tuple_list= [sup_date_time_tuple_list[i] for i in indices]

        for tuple in date_time_tuple_list:
            df_store_info_fp= pd.DataFrame(columns=["name", "start time", "audio file name", "present/absent (1/0)", "date", "time"])
            date= str(tuple[0])
            time_code= tuple[1]
            time_code= str(int(time_code))
            array_start_time, out_filenames = fetch_files(dir, date, time_code, bird, dir_tp_files, split)
            if len(array_start_time)==0:
                print (tuple)
            df_store_info_fp["audio file name"]= out_filenames
            df_store_info_fp["start time"]= array_start_time
            df_store_info_fp["date"]= date
            df_store_info_fp["time"]= time_code
            df_list_2.append(df_store_info_fp)
        df_store_info= pd.concat(df_list_2, ignore_index= True).copy()
        df_store_info["name"]= bird
        df_list.append(df_store_info)

    df= pd.concat(df_list, ignore_index= True)
    os.chdir(code)
    df.to_csv("false_negative_datasheet.csv")

    print("FN DS DONE YAY")
    return None

storage= "d:\\Research\\analyze_birdnet\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage#+ "common_resources\\" #refer to common resources folder in github
dir= common_resources+ "store_dataset_start_times\\"

time_interval=5 
conf=0.3
new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf)
# os.chdir(common_resources)
# df_datasheet= pd.read_excel("current_training_dataset.xlsx")
# df_datasheet= sd.add_date_time_codes(df_datasheet)
# sd.parse_datasheet_alldates_into_files(df_datasheet, time_interval, per_date_birdnet_only, datasheet_per_date_birdnet_only, common_resources)

                                       

# split= storage+ "audio_files_split\\"
# confusion_codes= ["tp", "fp", "fn"]

# # for confusion_code in confusion_codes:
# #     gcd.mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf, confusion_code)
datasheet_per_date_birdnet_only= "D:\\Research\\analyze_birdnet\\datasheet_per_date_birdnet_only\\"
split= storage+ "audio_files_split\\"
confusion_type= "fn"
mass_run_fetch_files(confusion_type, conf, code, split, dir)