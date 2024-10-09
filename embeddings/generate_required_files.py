import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

def make_dir(new_dir): 
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

#goal is to generate a list of bird names across the whole dataset.


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
                # print(date_array[iterant])
                continue
            else:
                # print(time_array[iterant])
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


# df_datasheet= pd.read_csv("current_training_dataset.csv")

def parse_df_for_seen_only(df, dir_to_store, indices):
    array_names, array_dates, array_timecodes, array_loc, array_wea, array_datetime = [], [], [], [], [], []
    array= [array_names, array_dates, array_timecodes, array_loc, array_wea, array_datetime]
    keys=["name", "date", "time code", "loc", "wea", "datetimelabel"]
    if "h" in indices:
        for iterant in range(len(df["Names"])):

            if str(np.array(df["No. of individuals heard"])[iterant]).isdigit() and str(np.array(df["No. of individuals heard"])[iterant])!="0":
                array_names.append(np.array(df["Names"])[iterant])
                array_dates.append(np.array(df["date code"])[iterant])
                array_timecodes.append(np.array(df["codes"])[iterant])
                array_loc.append(np.array(df["Location"])[iterant])
                array_wea.append(np.array(df["Weather"])[iterant])
                array_datetime.append(str(np.array(df["date code"])[iterant])+"_"+str(np.array(df["codes"])[iterant]))
    if "s" in indices:
        for iterant in range(len(df["Names"])):  
            if str(np.array(df["No. of individuals seen"])[iterant]).isdigit() and str(np.array(df["No. of individuals seen"])[iterant])!="0":
                array_names.append(np.array(df["Names"])[iterant])
                array_dates.append(np.array(df["date code"])[iterant])
                array_timecodes.append(np.array(df["codes"])[iterant])
                array_loc.append(np.array(df["Location"])[iterant])
                array_wea.append(np.array(df["Weather"])[iterant])
                array_datetime.append(str(np.array(df["date code"])[iterant])+"_"+str(np.array(df["codes"])[iterant]))
    if "hs" in indices:# or "sh" in indices:
        for iterant in range(len(df["Names"])):
            if str(np.array(df["No. of individuals seen and heard"])[iterant]).isdigit() and str(np.array(df["No. of individuals seen and heard"])[iterant])!="0":
                array_names.append(np.array(df["Names"])[iterant])
                array_dates.append(np.array(df["date code"])[iterant])
                array_timecodes.append(np.array(df["codes"])[iterant])
                array_loc.append(np.array(df["Location"])[iterant])
                array_wea.append(np.array(df["Weather"])[iterant])
                array_datetime.append(str(np.array(df["date code"])[iterant])+"_"+str(np.array(df["codes"])[iterant]))

    d= dict(zip(keys, array))
    new_df= pd.DataFrame.from_dict(d)
    os.chdir(dir_to_store)
    new_df.to_csv("filt_ds.csv")
    return new_df


# execute code
data_labels= ["seen"]
wd= "D:\\Research\\analyze_embeddings\\"
common_resources= wd+ "common_resources\\"
os.chdir(common_resources)
df_datasheet= pd.read_excel("current_training_dataset.xlsx")
df_datasheet["codes"]= add_time_codes_to_df(df_datasheet)
df_datasheet= add_date_time_codes(df_datasheet)
os.chdir(wd)
for label in data_labels:
    dir_to_store= common_resources + label +"_datafiles\\"
    make_dir(dir_to_store)
    if label== "heard":
        indices= ["h", "hs"]
    if label== "seen":
        indices= ["s"]
    parse_df_for_seen_only(df_datasheet, dir_to_store, indices)

    os.chdir(dir_to_store)
    df= pd.read_csv("filt_ds.csv")
    keys= list(set(df["datetimelabel"])) #unique labels
    groupbykeys= df.groupby("datetimelabel")
    output_arr=[]
    unique_bird_array= sorted(list(set(df["name"])))
    # create dict with value as count and key as bird; sort by values
    lis= list(df["name"])
    list_of_counts= []
    # list_of_birds= []
    for bird in unique_bird_array:
        occurrence = {item: lis.count(item) for item in lis}
        count= occurrence.get(bird)
        list_of_counts.append(count)
    dict_of_counts= dict(zip(unique_bird_array, list_of_counts))
    sorted_dict_of_counts= {k: v for k, v in sorted(dict_of_counts.items(), key=lambda item: item[1])}
    sorted_unique_bird_array= list(sorted_dict_of_counts.keys())
    y_arr= []
    loc_labels= []
    for key in keys:
        arr_1_0= np.zeros(shape= (len(sorted_unique_bird_array)))
        group_for_key= groupbykeys.get_group(key)
        output= list(set(group_for_key["name"]))
        loc= list(set(group_for_key["loc"]))[0]
        output_arr.append(output)
        loc_labels.append(loc)
        for it in range(len(sorted_unique_bird_array)):
            bird= sorted_unique_bird_array[it]
            if bird in output:
                arr_1_0[it]=1
        y_arr.append(arr_1_0)
    print(sorted_unique_bird_array)
    dictionary_output= dict(zip(keys, output_arr))
    dictionary_loclabels= dict(zip(keys, loc_labels))
    dictionary_y= dict(zip(keys, y_arr))
    make_dir(dir_to_store)
    os.chdir(dir_to_store)
    with open("output.pickle", 'wb') as handle:
        pickle.dump(dictionary_output, handle, protocol=-1)
    with open("loclabels.pickle", 'wb') as handle:
        pickle.dump(dictionary_loclabels, handle, protocol=-1)
    with open("y_arr.pickle", 'wb') as handle:
        pickle.dump(dictionary_y, handle, protocol=-1)

    print("Task done for " + label)
        




