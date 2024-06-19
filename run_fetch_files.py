import os
import glob
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import pickle
import splicer as sp
import generate_confusion_data as gcd
import fetch_files_support_functions as ff

common_resources= "D:\\Research\\Rohini 2024\\Bird Data\\common_resources\\"
dir= common_resources+ "\\store_birdnet_info\\"

storage= "d:\\Research\\Rohini 2024\\Bird Data\\" 
time_interval=5 
conf=0.3
new_dir, code, split, datasheet_per_date_birdnet_only, per_date_birdnet_only= sp.create_required_directories(storage, time_interval, conf)
split= storage+ "audio_files_split\\"
confusion_codes= ["tp", "fp", "fn"]

for confusion_code in confusion_codes:
    gcd.mass_run_all_dates(code, datasheet_per_date_birdnet_only, per_date_birdnet_only, conf, confusion_code)
    
confusion_type= "fp"
ff.mass_run_fetch_files(confusion_type, conf, code, split)

