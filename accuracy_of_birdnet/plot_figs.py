import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
mlt.style.use("ggplot")
import seaborn as sns
import os
from splicer import make_dir
import splicer as sp

storage= "d:\\Research\\analyze_birdnet\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage#+ "common_resources\\" #refer to common resources folder in github

interval_of_conf= list(np.arange(0.3, 0.9, 0.02)) #input the ranges of conf to run birdnet over
interval_of_conf= [np.round(i,2) for i in interval_of_conf]
time_interval= 5 #in minutes
confusion_entries= ["tp", "fn", "fp"]

# list_of_birds= ["large billed crow","white cheeked barbet","rose ringed parakeet","oriental magpie robin","red whiskered bulbul"]
os.chdir(common_resources)
df_datasheet= pd.read_excel("current_training_dataset.xlsx")
list_of_birds= list(set(list(df_datasheet["Names"])))

df_list= []
for conf in interval_of_conf:
    new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf)
    for confusion in confusion_entries:
        dir= code+"\\stats_"+confusion+"\\"
        os.chdir(dir)
        df= pd.read_csv("all_dates_merged_w_names_"+str(conf)+"confidence.csv")
        df["confidence"]= conf
        df_list.append(df)

df= pd.concat(df_list, ignore_index= True)
# print(df)
df_list= []
groupy= df.groupby("name")
for it in range(len(list_of_birds)):
    bird= list_of_birds[it]
    try:
        current_df=  groupy.get_group(bird)
    except KeyError:
        continue
    new_df= pd.DataFrame(columns=["name", "datasheet", "confidence", "confusion", "confusion numbers"])
    array=["tp", "fp"]
    array_counts=[]
    array_label=[]
    array_confidence=[]
    for bt in range(len(array)):
        # print(current_df)
        group= current_df.groupby("confusion")
        # print(bird, array[bt])
        try:
            confusion_df= group.get_group(array[bt])
        except KeyError:
            for conf in interval_of_conf:
                array_label.append(array[bt])
                array_confidence.append(conf)
                array_counts.append(0)
            continue
        for conf in interval_of_conf:
            # print(confusion_df)
            try:
                print(array[bt])
                conf_df= confusion_df.groupby("confidence").get_group(conf)
                array_counts.append(len(list(conf_df["time code"])))
            except KeyError:
                array_counts.append(0)
            array_label.append(array[bt])
            array_confidence.append(conf)

    new_df["confusion matrix entries"]=array_label
    new_df["counts"]=array_counts
    new_df["confidence"]=array_confidence
    new_df["name"]=bird
    df_list.append(new_df)
    print(new_df)
    group= new_df.groupby("confusion matrix entries")
    fp_confusion= group.get_group("fp")
    tp_confusion= group.get_group("tp")

    fig= plt.figure(figsize=(16,9))
    plt.plot(fp_confusion["confidence"],fp_confusion["counts"], label="false positive")
    plt.plot(tp_confusion["confidence"],tp_confusion["counts"], label="true positive")
    plt.title(bird+ " line plot")
    plt.legend()
    wd= os.getcwd()
    dir_store_plots= storage +"\\plots_storage_confusion_0.3_to_0.9_step_0.02\\tp_fp_all_birds_line_plots\\"
    make_dir(dir_store_plots)
    os.chdir(dir_store_plots)
    plt.savefig(bird+"_confusion_0.3_to_0.9_tp_fp_line_plots.png", bbox_inches= "tight")
    os.chdir(wd)
    plt.clf()

    # fig= plt.figure(figsize=(16,9))
    # # g = sns.catplot(
    # # data=new_df, kind="bar",
    # # x="confusion", y="confusion numbers", hue="confidence", palette="dark", alpha=.6, height=4
    # # )
    # # g.despine(left=True)
    # # g.set_axis_labels("", "counts")
    # # g.legend.set_title("confidence")
    # # for i in g.containers:
    # #     g.bar_label(i,)
    # ax= sns.barplot( data=new_df, x="confusion matrix entries", y="counts", hue="confidence", palette="dark", alpha=.6)
    # for i in ax.containers:
    #     ax.bar_label(i,)
    # plt.title(bird)
    # wd= os.getcwd()
    # dir_store_plots= storage +"\\plots_storage_confusion_0.3_to_0.9_step_0.02\\tp_only_all_birds\\"
    # make_dir(dir_store_plots)
    # os.chdir(dir_store_plots)
    # plt.savefig(bird+"_confusion_0.3_to_0.9_tp_only.png", bbox_inches= "tight")
    # os.chdir(wd)
    # # plt.clf()


# current_df.set_index('confidence', inplace=True)
# current_df = current_df.stack().to_frame('value').reset_index()
# print(current_df)
# plt.show()
# dataframe= pd.concat(df_list, ignore_index= True)

# #plot line plot for tp and fp
# confusion_entries= ["tp", "fp"]
# print(list_of_birds)
# for bird in list_of_birds:
#     group= dataframe.groupby("name")

#     try:
#         df_confusion_bird= group.get_group(bird)
#     except KeyError:
#         continue
    
#     group= dataframe.groupby("confusion matrix entries")
#     fp_confusion= group.get_group("fp")
#     tp_confusion= group.get_group("tp")
#     # print(list(df_confusion["counts"]))
#     fig= plt.figure(figsize=(16,9))
#     plt.plot(fp_confusion["confidence"],fp_confusion["counts"], label=confusion)
#     plt.plot(tp_confusion["confidence"],tp_confusion["counts"], label=confusion)
#     plt.title(bird+ " line plot")
#     plt.legend()
#     wd= os.getcwd()
#     dir_store_plots= storage +"\\plots_storage_confusion_0.3_to_0.9_step_0.02\\tp_fp_all_birds_line_plots\\"
#     make_dir(dir_store_plots)
#     os.chdir(dir_store_plots)
#     plt.savefig(bird+"_confusion_0.3_to_0.9_tp_fp_line_plots.png", bbox_inches= "tight")
#     os.chdir(wd)
#     plt.clf()
    


