import os
import pandas as pd
import numpy as np

common_resources= os.getcwd()

def create_date_loc_df(df_datasheet):

    array_dates= df_datasheet["Date"]
    array_dates_unique= list(set(array_dates))
    array_indices= []
    for i in range(len(array_dates_unique)):
        for j in range(len(array_dates)):
            if array_dates_unique[i]==array_dates[j]:
                array_indices.append(j)
                break

    list_w_areanw=[]
    for it in range(len(array_indices)):
        date= df_datasheet["Date"][array_indices[it]]
        date= str(date)[0:4]+str(date)[5:7]+str(date)[8:10]
        loc=df_datasheet["Location"][array_indices[it]]

        if loc=="Jubilee":
            lat, lon= 13.022551345783762, 77.57019815297808
        if loc=="B7":
            lat, lon= 13.020562422900062, 77.56575334233537
        if loc=="Rocks":
            lat, lon= 13.021233810004006, 77.56969862313451

        list_w_areanw.append([date, df_datasheet["Location"][array_indices[it]], df_datasheet["Weather"][array_indices[it]], lat, lon])
    list_w_areanw=np.array(list_w_areanw)
    
    df_date_loc_info= pd.DataFrame(columns=["date", "Location", "Weather"])
    df_date_loc_info["date"]= list_w_areanw[:,0]
    df_date_loc_info["Location"]= list_w_areanw[:,1]
    df_date_loc_info["Weather"]= list_w_areanw[:,2]
    df_date_loc_info["lat"]= list_w_areanw[:,3]
    df_date_loc_info["lon"]= list_w_areanw[:,4]

    os.chdir(common_resources)
    df_date_loc_info.to_csv("date_loc_df.csv")
    return df_date_loc_info

os.chdir(common_resources)
df_datasheet= pd.read_excel("current_training.xlsx")
create_date_loc_df(df_datasheet)

array= df_datasheet["Date"]
array= [str(i)[0:4]+str(i)[5:7]+str(i)[8:10] for i in array]
time_array=df_datasheet["Time"]
time_array=[str(i)[3:5] for i in time_array]

df_datasheet["date code"]= array
df_datasheet["time code"]= time_array

df_datasheet.to_csv("current_training_w_codes.csv")