import os
import numpy as np
import pandas as pd

#create directories
def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

#given two arrays with species, find their union
def find_unique_species_in_union(datasheet_merged, bn_merged):
    array_ds= list(datasheet_merged["common name"])
    array_bn= list(bn_merged["common name"])
    set_ds= set(array_ds)
    set_bn= set(array_bn)
    union= list(set_ds.union(set_bn))
    return union

#generates stats for the confusion matrix for each species over the whole dataset (ground truth + birdnet)
def generate_confusion_matrix_info(time_interval, conf_threshold_for_bn, datasheet_per_date_birdnet_only, per_date_birdnet_only, dir_store_confusion):
    os.chdir(datasheet_per_date_birdnet_only)
    datasheet_merged= pd.read_csv(str(time_interval)+"mintask_merged_datasheet.csv")
    os.chdir(per_date_birdnet_only)
    bn_merged= pd.read_csv("merged_mass_birdnet_"+str(time_interval)+"mins.csv")

    df_confusion= pd.DataFrame(columns=["common name", "datasheet", "false negative", "true positive", "false positive", "birdnet", "union"])
    df_confusion["common name"]= find_unique_species_in_union(datasheet_merged, bn_merged)

    time_date_tuples_univ=[]
    for jt in range(len(datasheet_merged["common name"])):
        time_date_tuples_univ.append((np.array(datasheet_merged["time code"])[jt], np.array(datasheet_merged["date"])[jt]))

    for it in range(len(df_confusion["common name"])):
        name= df_confusion["common name"][it]
        ds=list(datasheet_merged["common name"]).count(name)
        bn=list(bn_merged["common name"]).count(name)
        df_confusion["datasheet"][it]= ds
        df_confusion["birdnet"][it]= bn

        # grouped_ds_by_name= datasheet_merged.groupby("common name").get_group(name)
        try:
            grouped_bn_by_name= bn_merged.groupby("common name").get_group(name)
            bn_time_date_tuples=[]
            for jt in range(len(grouped_bn_by_name["common name"])):
                bn_time_date_tuples.append((np.array(grouped_bn_by_name["time"])[jt], np.array(grouped_bn_by_name["date"])[jt]))
        except KeyError:
            bn_time_date_tuples=[]

        try:
            grouped_ds_by_name= datasheet_merged.groupby("common name").get_group(name)
            ds_time_date_tuples=[]
            for jt in range(len(grouped_ds_by_name["common name"])):
                ds_time_date_tuples.append((np.array(grouped_ds_by_name["time code"])[jt], np.array(grouped_ds_by_name["date"])[jt]))
        except KeyError:
            ds_time_date_tuples=[]

        #intersection
        true_positive= [tuples for tuples in ds_time_date_tuples if tuples in bn_time_date_tuples]
        tp= len(true_positive)
        #bn\ds
        false_positive= [tuples for tuples in bn_time_date_tuples if tuples not in ds_time_date_tuples]
        fp= len(false_positive)
        #ds\bn
        false_negative= [tuples for tuples in ds_time_date_tuples if tuples not in bn_time_date_tuples]
        fn= len(false_negative)
        #tn
        union= ds + bn - tp
        # universal= len(time_date_tuples_univ) #both for ds and bn hence twice
        # tn= universal- union

        df_confusion["true positive"][it]= tp
        df_confusion["false positive"][it]= fp
        df_confusion["false negative"][it]= fn
        # df_confusion["true negative"][it]= tn
        df_confusion["union"][it]= union

    make_dir(dir_store_confusion)
    os.chdir(dir_store_confusion)
    df_confusion.to_csv(str(time_interval)+"min_"+str(conf_threshold_for_bn)+"conf_confusion_info.csv")
    return "successfully confused :D for" + str(conf_threshold_for_bn)+"confidence"