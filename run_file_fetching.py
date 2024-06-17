#importing python files
from basic_codes import splicer as sp
from basic_codes import supporting_functions_for_audio_fetch as af


storage= "d:\\Research\\Rohini 2024\\Bird Data\\" #source folder
sound_data= storage+ "sound_data\\" #location of sound files
common_resources= storage+ "common_resources\\" #refer to common resources folder in github

time_interval= 5 #in minutes
conf_threshold_for_bns= [0.3, 0.6, 0.9] #ranges from 0 to 1
confusion = ["tp","fn","fp"]

for conf_threshold_for_bn in conf_threshold_for_bns:
    # conf_threshold_for_bn= conf_threshold_for_bns[it]
    new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf_threshold_for_bn)
    split= "D:\\Research\\Rohini 2024\\Bird Data\\audio_files_split\\"
    for confusion_entry in confusion:
        samples_audio, big_dir= af.run_required_code(confusion_entry, code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf_threshold_for_bn, split, delete_files=False)
        # af.get_piece_of_audio(date_folder, audio_file, start_time, end_time, bird, store_fp_pieces_dir, kv)
        af.run_birdnet_and_fetch_files(samples_audio, big_dir, common_resources, conf_threshold_for_bn)