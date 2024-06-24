import os
import glob
import numpy as np
import pandas as pd
import shutil
import run_birdnet as rb

# Load and initialize the BirdNET-Analyzer models.
# analyzer = Analyzer()


def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

def merge_bds(per_date_birdnet_only, date):
    os.chdir(per_date_birdnet_only)
    df_list= []
    list_of_csvs= glob.glob(date+"*")

    for csv in list_of_csvs:
        df= pd.read_csv(csv)
        df_list.append(df)
    dataframe= pd.concat(df_list, ignore_index= True)

    dataframe.to_csv(date+"_split_datasheet5.csv")
def mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf, confusion_code):
    #given

    os.chdir(datasheet_per_date_birdnet_only)
    file_list= glob.glob("*.csv")
    dates= [i[0:8] for i in file_list]
    dates=dates[:-1]

    super_stats_df=pd.DataFrame(columns=["date", "name", "time code" ])

    for iterant in range(len(dates)):
        date= dates[iterant]
        merge_bds(per_date_birdnet_only, date)

        os.chdir(datasheet_per_date_birdnet_only)
        datasheet_split = glob.glob(date+"*")
        df_ds_split= pd.read_csv(datasheet_split[0])

        os.chdir(per_date_birdnet_only)
        df_bd_split= pd.read_csv(date+"_split_datasheet5.csv")

        array_times= df_ds_split['time code']
        array_times=[str(i) for i in array_times]
        array_times=["0"+i if len(i)<2 else i for i in array_times]
        df_ds_split['time code']= array_times
        df_ds_split
        array_times= df_bd_split['time']
        array_times=[str(i) for i in array_times]
        array_times=["0"+i[1] if i[0]=="_" else i for i in array_times]
        df_bd_split['time']= array_times

        df_bd_split["common name"]=rb.remove_special_from_names(df_bd_split["common name"])
        # print(df_bd_split)


        date= dates[iterant]

        stats_df= pd.DataFrame(columns=["date","name", "time code"])

        input_time_interval=5

        time_codes= np.arange(0, 55, input_time_interval)

        time_codes=[str(i) for i in time_codes]
        time_codes=["0"+i if len(i)<2 else i for i in time_codes]

        # bd_minus_ds_array=[]

        # print(date)
        for it in range(len(time_codes)):
            print(date)
            print(df_ds_split)
            time= time_codes[it]
            new_df= pd.DataFrame(columns=["date","name", "time code"])
            ds_groupby= df_ds_split.groupby("time code")
            try:
                ds_compare= ds_groupby.get_group(time_codes[it])
                ds_compare_array=set(np.array(ds_compare["common name"]))
            except KeyError:
                ds_compare_array=set([])

            bd_groupby= df_bd_split.groupby("time")
            # print(df_bd_split)
            try:
                bd_compare= bd_groupby.get_group(time_codes[it])
                bd_compare_array=set(np.array(bd_compare["common name"]))
            except KeyError:
                bd_compare_array=set([])
            
            print(time_codes[it], bd_compare_array, ds_compare_array, date)
            if confusion_code== "tp":
                bd_intersection_ds= bd_compare_array.copy()
                bd_intersection_ds= bd_intersection_ds.intersection(ds_compare_array)
                new_df["name"]= list(bd_intersection_ds)
                new_df["confusion"]= "tp"
            if confusion_code== "fp":
                bd_minus_ds= bd_compare_array.copy()
                bd_minus_ds= bd_minus_ds.difference(ds_compare_array)
                new_df["name"]= list(bd_minus_ds)
                new_df["confusion"]= "fp"
            if confusion_code== "fn":
                ds_minus_bd= ds_compare_array.copy()
                ds_minus_bd= ds_minus_bd.difference(bd_compare_array)
                new_df["name"]= list(ds_minus_bd)
                new_df["confusion"]= "fn"

            new_df["date"]= date
            new_df["time code"]=time
            stats_df= pd.concat([stats_df, new_df])

        stats_df["date"]= date
        # stats_df.to_csv(date+"_stats_w_names_cl.csv")  
        super_stats_df= pd.concat([super_stats_df, stats_df])

    dir= code+"\\stats_"+confusion_code+"\\"
    make_dir(dir)
    os.chdir(dir)
    super_stats_df.to_csv("all_dates_merged_w_names_"+str(conf)+"confidence.csv")
    return "done :D"