import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

def make_dir(new_dir): 
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

#goal is to generate a list of bird names across the whole dataset.
wd= "D:\\github\\man-vs-BirdNet-summer-2024-\\embeddings"
common_resources= wd+ "\\common_resources\\"
os.chdir(common_resources)
df_datasheet= pd.read_excel("current_training_dataset.xlsx")
os.chdir(wd)
print(df_datasheet)

def find_difference_in_minutes(time1, time2):
    t1= datetime.strptime(str(time1), "%H:%M:%S")
    t2= datetime.strptime(str(time2), "%H:%M:%S")
    difference= t2-t1
    minutes= difference.total_seconds()/60
    return minutes

def add_time_codes_to_df(df):
    time_code= 0
    code_array= []
    date_array= np.array(df["Date"])
    time_array= np.array(df["Time"])

    for iterant in range(len(df["Time"])):
        if date_array[iterant-1]== date_array[iterant]:
            if time_array[iterant]==np.nan:
                print(date_array[iterant])
                continue
            else:
                print(time_array[iterant])
                minute_diff= find_difference_in_minutes(time_array[iterant-1], time_array[iterant])
                if minute_diff==0:
                    time_code=time_code
                else:
                    time_code+= int(minute_diff)
        else:
            time_code=0
        
        code_array.append(time_code)
    return code_array

def add_date_time_codes(df_datasheet):
    array= df_datasheet["Date"]
    # print(array[0])
    array= [int(str(i)[0:4]+str(i)[5:7]+str(i)[8:10]) for i in array]
    time_array=df_datasheet["codes"]
    time_array=[str(i) for i in time_array]
    time_array=["0"+i if len(i)<2 else i for i in time_array]
    df_datasheet["date code"]= array
    df_datasheet["time code"]= time_array
    return df_datasheet

df_datasheet["codes"]= add_time_codes_to_df(df_datasheet)
df_datasheet= add_date_time_codes(df_datasheet)
# df_datasheet= pd.read_csv("current_training_dataset.csv")

def parse_df_for_seen_only(df, dir_to_store):
    array_names, array_dates, array_timecodes, array_loc, array_wea = [], [], [], [], []
    array= [array_names, array_dates, array_timecodes, array_loc, array_wea]
    keys=["name", "date", "time code", "loc", "wea"]

    for iterant in range(len(df["Names"])):
        if str(np.array(df["No. of individuals seen"])[iterant]).isdigit() and str(np.array(df["No. of individuals seen"])[iterant])!="0":
            array_names.append(np.array(df["Names"])[iterant])
            array_dates.append(np.array(df["date code"])[iterant])
            array_timecodes.append(np.array(df["codes"])[iterant])
            array_loc.append(np.array(df["Location"])[iterant])
            array_wea.append(np.array(df["Weather"])[iterant])

        if str(np.array(df["No. of individuals seen and heard"])[iterant]).isdigit() and str(np.array(df["No. of individuals seen and heard"])[iterant])!="0":
            array_names.append(np.array(df["Names"])[iterant])
            array_dates.append(np.array(df["date code"])[iterant])
            array_timecodes.append(np.array(df["codes"])[iterant])
            array_loc.append(np.array(df["Location"])[iterant])
            array_wea.append(np.array(df["Weather"])[iterant])

    d= dict(zip(keys, array))
    new_df= pd.DataFrame.from_dict(d)
    os.chdir(dir_to_store)
    new_df.to_csv("filt_ds.csv")
    return new_df

parse_df_for_seen_only(df_datasheet, common_resources)


        
        

