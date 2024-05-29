import soundfile as sf
import librosa
import os
import glob

#create new directories
def make_dir(new_dir): 
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return "made dir " + str(new_dir)

#generate all the directories required to run the code
def create_required_directories(main_dir, time_interval, conf_threshold_for_bn): 
    new_dir= main_dir+ "folder_for_" +str(time_interval)+"mins_"+str(conf_threshold_for_bn)+"confidence\\"
    code= new_dir +"code\\"
    split= new_dir+ "audio_files_split\\"
    datasheet_per_date_birdnet_only = code+"datasheet_per_date_birdnet_only\\"
    per_date_birdnet_only= code+"per_date_birdnet_only\\"

    make_dir(new_dir)
    make_dir(code)
    make_dir(split)
    make_dir(datasheet_per_date_birdnet_only)
    make_dir(per_date_birdnet_only)

    return new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only

#parse the audio folder for all the dates and shortlist based on ground truth data available
def generate_filenames_from_dataset(df, sound_folder):
    os.chdir(sound_folder)
    list_of_available_files= glob.glob("*.WAV")

    number_of_entries= len(df["Date"])
    available_file_count= len(list_of_available_files)
    array_w_common_elements=[]
    dupe_array=[]

    list_of_dates_from_files=[i[0:8] for i in list_of_available_files]

    for it in range(number_of_entries):
        element= df["Date"][it]
        filename= str(element)[0:4]+str(element)[5:7]+str(element)[8:10]
        for jt in range(available_file_count):

            if filename == list_of_dates_from_files[jt]:
                if filename not in dupe_array:
                    dupe_array.append(filename)
                    array_w_common_elements.append(list_of_available_files[jt])
    print("List of dates you are running the code on: ")
    print(list_of_dates_from_files)
    return array_w_common_elements

#generate slices of audio of desired time interval for a given audio file
def slicer(filename, time_interval, sound_folder, split): 
    os.chdir(sound_folder)
    audio, sr = librosa.load(filename)

    # Get number of samples, time_interval in minutes
    buffer = time_interval * sr * 60 #the 60 is to convert minutes to seconds

    samples_total = len(audio)
    samples_wrote = 0
    counter = 0
    old_directory= os.getcwd()
    new_path= split+filename[0:8]+"\\"
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    os.chdir(new_path)
    while samples_wrote < samples_total:
        #check if the buffer is not exceeding total samples 
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        block = audio[samples_wrote : (samples_wrote + buffer)]
        out_filename =  filename[0:8] + "_" + "split_" + str(counter) +".WAV"

        # Write 2 second segment
        sf.write(out_filename, block, sr)
        counter += int(time_interval)
        samples_wrote += buffer
    os.chdir(old_directory)
    return "Done"

#run slicer on all audio files
def run_slicer_on_all_files(df, sound_folder, time_interval, split):

    filenames_w_data= generate_filenames_from_dataset(df, sound_folder)

    files_processed_count= len(filenames_w_data)

    for it in range(files_processed_count):
        filename= filenames_w_data[it]
        slicer(filename, time_interval, sound_folder, split)
        print("Done_for_"+filename)
    
    print(str(files_processed_count) +" files processed")