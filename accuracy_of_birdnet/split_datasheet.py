import os
import glob
import numpy as np
import pandas as pd
import datetime
import pickle

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)
#to ground truth add date and time codes to simplify comparison
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

def find_difference_in_minutes(time1, time2):
    t1= datetime.strptime(str(time1), "%H:%M:%S")
    t2= datetime.strptime(str(time2), "%H:%M:%S")
    difference= t2-t1
    minutes= difference.total_seconds()/60
    return minutes

def add_required_start_time_codes(df_datasheet):
    start_time_array= []
    for entry in range(len(df_datasheet["Time"])):
        # variable= df_datasheet["Time"][entry]
        # print(type(df_datasheet["Time"][entry]), isinstance(variable, datetime.time))
        time_code= df_datasheet["codes"][entry]%5
        start_time= time_code*60
        start_time_array.append(start_time)
    # print(start_time_array)
    df_datasheet["start time"]= start_time_array
    return df_datasheet

#generate a subsection of ds given time and date
def parse_datasheet_per_time_per_date(df_datasheet, date, input_time_code, input_time_interval, common_resources):
    
    #find birds that are h or hs during interval 25
    groupby_data= df_datasheet.groupby("date code")
    # print(df_datasheet)
    # print(set(df_datasheet["date code"]))
    # print(df_datasheet)

    df_given_date= groupby_data.get_group(int(date))

    # print(df_given_date)

    groupby_df_given_date= df_given_date.groupby("time code")

    # input_time_interval= 5 #minutes
    # input_time_code="05"

    range_of_times=np.arange(int(input_time_code), int(input_time_code)+input_time_interval)
    range_of_times=[str(i) for i in range_of_times]
    range_of_times=["0"+i if len(i)<2 else i for i in range_of_times ]
    range_of_times

    #concat groups in range of times
    merge_time_group= pd.DataFrame(columns=["Location",	"Weather", "Temperature","Date","Time","Species","No. of individuals seen","No. of individuals heard","No. of individuals seen and heard","Comments","Names","date code","time code", "start time"])
    for it in range(input_time_interval):
        try:
            new_time_group= groupby_df_given_date.get_group(str(range_of_times[it]))
            # print(new_time_group)
            merge_time_group= pd.concat([merge_time_group, new_time_group])
        except KeyError:
            merge_time_group=merge_time_group

    merge_time_group #this is the data set to work on now!
    # print(merge_time_group)
    df_to_store_data_at_time_step= pd.DataFrame(columns=["common name", "time code","date"])
    array_to_store=[]
    store_startime=[]
    for it in range(len(merge_time_group["No. of individuals heard"])):
        if str(np.array(merge_time_group["No. of individuals heard"])[it]).isdigit() and str(np.array(merge_time_group["No. of individuals heard"])[it])!="0":
            array_to_store.append(np.array(merge_time_group["Names"])[it])
            store_startime.append(np.array(merge_time_group["start time"])[it])
        if str(np.array(merge_time_group["No. of individuals seen and heard"])[it]).isdigit() and str(np.array(merge_time_group["No. of individuals seen and heard"])[it])!="0":
            array_to_store.append(np.array(merge_time_group["Names"])[it])
            store_startime.append(np.array(merge_time_group["start time"])[it])
    # print(array_to_store)
    unique_to_store_for_time= list(set(array_to_store))

    dir_to_store= common_resources+ "store_dataset_start_times\\"
    make_dir(dir_to_store)
    keys= []
    array_start_times= []
    
    for bird in unique_to_store_for_time:
        keys.append(bird)
        bird_start_time=[]
        for it in range(len(array_to_store)):
            if bird==array_to_store[it]:
                bird_start_time.append(store_startime[it])
        array_start_times.append(list(set(bird_start_time)))
    
    dictionary= dict(zip(keys, array_start_times))
    print(dictionary)
    os.chdir(dir_to_store)
    with open(str(date)+ "_split_" +str(input_time_code)+ ".pickle", 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=-1)

    df_to_store_data_at_time_step["common name"]= unique_to_store_for_time
    df_to_store_data_at_time_step["time code"]=input_time_code
    df_to_store_data_at_time_step["date"]=date

    return df_to_store_data_at_time_step

#generate a subsection of ds given date for all times in that date
def parse_datasheet_per_date(df_datasheet, date, input_time_interval, common_resources):
    time_codes= np.arange(0, 60, input_time_interval)

    time_codes=[str(i) for i in time_codes]
    time_codes=["0"+i if len(i)<2 else i for i in time_codes]
    merged_df= pd.DataFrame(columns=["common name", "time code","date"])
    for it in range(len(time_codes)):
        current_df= parse_datasheet_per_time_per_date(df_datasheet, date, time_codes[it], 5, common_resources)
        merged_df= pd.concat([merged_df, current_df])

    return merged_df

#generate a subsection of ds for all dates and all times in any date

def parse_datasheet_alldates_into_files(df_datasheet, input_time_interval, per_date_birdnet_only, datasheet_per_date_birdnet_only, common_resources):
    df_datasheet= add_required_start_time_codes(df_datasheet)
    os.chdir(per_date_birdnet_only)
    print(os.getcwd())
    csv_files= glob.glob("*.csv")
    csv_files= csv_files[:-1]
    list_w_no_data=[]
    list_of_dates_to_run= [int(i[0:8]) for i in csv_files]
    # list_of_dates_to_run= [i for i in list_of_dates_to_run if i!= 20230530 and i!=20230906 and i!=20231115 and i!=20231207 and i!= 20240411 and i!=20240223 and i!=20240111]
    # print(list_of_dates_to_run)
    super_df= pd.DataFrame(columns=["common name", "time code","date"])
    for it in range(len(list_of_dates_to_run)):
        try:
            current_df= parse_datasheet_per_date(df_datasheet, list_of_dates_to_run[it], input_time_interval, common_resources)
        except KeyError:
            # print(list_of_dates_to_run[it])
            list_w_no_data.append(list_of_dates_to_run[it])
            continue
        # print(current_df)
        os.chdir(datasheet_per_date_birdnet_only)
        current_df.to_csv(str(list_of_dates_to_run[it])+"_split_datasheet"+str(input_time_interval)+".csv")
        super_df= pd.concat([super_df, current_df])
        # print("done for "+ str(list_of_dates_to_run[it]))

    os.chdir(datasheet_per_date_birdnet_only)
    super_df.to_csv(str(input_time_interval)+"mintask_merged_datasheet.csv")
    print(set(list_w_no_data))
    return "done"