import os
import glob
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import pickle


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

def get_start_time_info_per_bird_per_timecode(dir, date, time_code, bird):
    os.chdir(dir)
    date_folder= glob.glob(date+"*")[0]
    os.chdir(dir+ date_folder)
    
    file_name= glob.glob("*"+time_code+".pickle")[0]
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    try:
        array_start_time= data[bird]
    except KeyError:
        array_start_time= []
    return array_start_time

def get_piece_of_audio(in_folder, audio_file, start_time, bird, dir_to_store, split):
    # os.chdir(sound_folder)
    os.chdir(split)
    os.chdir(in_folder)
    audio, sr = librosa.load(audio_file)

    end_time= start_time+3
    slice= audio[int(start_time)*sr: int(end_time)*sr]
    out_filename =  audio_file[:-4] +"_"+str(start_time) +"_piece_of_"+str(bird)+ ".WAV"

    os.chdir(dir_to_store)
    sf.write(out_filename, slice, sr)
    print("pieced "+ bird)
    return out_filename

def fetch_files(dir, date, time_code, bird, dir_tp_files, split):
    array_start_time= get_start_time_info_per_bird_per_timecode(dir, date, time_code, bird)
    dir_to_store= dir_tp_files + bird+ "\\"
    make_dir(dir_to_store)
    #find date folder in audio split
    os.chdir(split)
    input_folder_name= glob.glob(date+"*")[0]
    print(input_folder_name)
    audio_file_name= date+"_split_"+time_code+".WAV"
    if len(array_start_time)==0:
        return array_start_time
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
        date_time_tuple_list= [(date_list[i],time_list[i]) for i in range(len(bird_df["name"]))]
        df_list_2= []
        for tuple in date_time_tuple_list:
            df_store_info_fp= pd.DataFrame(columns=["name", "start time", "audio file name", "present/absent (1/0)", "date", "time"])
            date= str(tuple[0])
            time_code= tuple[1]
            time_code= str(int(time_code))
            array_start_time, out_filenames = fetch_files(dir, date, time_code, bird, dir_tp_files, split)
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
    df.to_csv("false_positive_datasheet.csv")

    print("FP DS DONE YAY")
    return None


