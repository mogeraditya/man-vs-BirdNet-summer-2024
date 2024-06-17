import os
import glob
import numpy as np
import pandas as pd


#to ground truth add date and time codes to simplify comparison
def add_date_time_codes(df_datasheet):
    array= df_datasheet["Date"]
    array= [int(str(i)[6:10]+str(i)[3:5]+str(i)[0:2]) for i in array]
    time_array=df_datasheet["Time"]
    time_array=[str(i)[3:5] for i in time_array]
    df_datasheet["date code"]= array
    df_datasheet["time code"]= time_array
    return df_datasheet

#generate a subsection of ds given time and date
def parse_datasheet_per_time_per_date(df_datasheet, date, input_time_code, input_time_interval):

    #find birds that are h or hs during interval 25
    groupby_data= df_datasheet.groupby("date code")
    # print(df_datasheet)
    df_given_date= groupby_data.get_group(date)
    # print(df_given_date)

    groupby_df_given_date= df_given_date.groupby("time code")

    # input_time_interval= 5 #minutes
    # input_time_code="05"

    range_of_times=np.arange(int(input_time_code), int(input_time_code)+input_time_interval)
    range_of_times=[str(i) for i in range_of_times]
    range_of_times=["0"+i if len(i)<2 else i for i in range_of_times ]
    range_of_times

    #concat groups in range of times
    merge_time_group= pd.DataFrame(columns=["Location",	"Weather", "Temperature","Date","Time","Species","No. of individuals seen","No. of individuals heard","No. of individuals seen and heard","Comments","Names","date code","time code"])
    for it in range(input_time_interval):
        try:
            new_time_group= groupby_df_given_date.get_group(range_of_times[it])
            # print(new_time_group)
            merge_time_group= pd.concat([merge_time_group, new_time_group])
        except KeyError:
            merge_time_group=merge_time_group

    merge_time_group #this is the data set to work on now!
    # print(merge_time_group)
    df_to_store_data_at_time_step= pd.DataFrame(columns=["common name", "time code","date"])
    array_to_store=[]
    for it in range(len(merge_time_group["No. of individuals heard"])):
        if np.array(merge_time_group["No. of individuals heard"])[it].isdigit() and np.array(merge_time_group["No. of individuals heard"])[it]!="0":
            array_to_store.append(np.array(merge_time_group["Names"])[it])
        if np.array(merge_time_group["No. of individuals seen and heard"])[it].isdigit() and np.array(merge_time_group["No. of individuals seen and heard"])[it]!="0":
            array_to_store.append(np.array(merge_time_group["Names"])[it])
    # print(array_to_store)
    unique_to_store_for_time= list(set(array_to_store))

    df_to_store_data_at_time_step["common name"]= unique_to_store_for_time
    df_to_store_data_at_time_step["time code"]=input_time_code
    df_to_store_data_at_time_step["date"]=date
    return df_to_store_data_at_time_step

#generate a subsection of ds given date for all times in that date
def parse_datasheet_per_date(df_datasheet, date, input_time_interval):
    time_codes= np.arange(0, 60, input_time_interval)

    time_codes=[str(i) for i in time_codes]
    time_codes=["0"+i if len(i)<2 else i for i in time_codes]
    merged_df= pd.DataFrame(columns=["common name", "time code","date"])
    for it in range(len(time_codes)):
        current_df= parse_datasheet_per_time_per_date(df_datasheet, date, time_codes[it], 5)
        merged_df= pd.concat([merged_df, current_df])

    return merged_df

#generate a subsection of ds for all dates and all times in any date
def parse_datasheet_alldates_into_files(df_datasheet, input_time_interval, per_date_birdnet_only, datasheet_per_date_birdnet_only):
    os.chdir(per_date_birdnet_only)
    csv_files= glob.glob("*.csv")
    csv_files= csv_files[:-1]

    list_of_dates_to_run= [int(i[0:8]) for i in csv_files]
    # print(list_of_dates_to_run)
    super_df= pd.DataFrame(columns=["common name", "time code","date"])
    for it in range(len(list_of_dates_to_run)):
        current_df= parse_datasheet_per_date(df_datasheet, list_of_dates_to_run[it], input_time_interval)
        # print(current_df)
        os.chdir(datasheet_per_date_birdnet_only)
        current_df.to_csv(str(list_of_dates_to_run[it])+"_split_datasheet"+str(input_time_interval)+".csv")
        super_df= pd.concat([super_df, current_df])

    os.chdir(datasheet_per_date_birdnet_only)
    super_df.to_csv(str(input_time_interval)+"mintask_merged_datasheet.csv")
    return "done"