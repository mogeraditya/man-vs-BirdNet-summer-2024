
import os
import glob
import pandas as pd

#importing python files
import splicer as sp
import run_birdnet as rb
import split_datasheet as sd
import analysis_of_data as aod

import soundfile as sf
import librosa
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import random
import matplotlib as mpl

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

import shutil
from splicer import make_dir
import run_birdnet as rb
import generate_false_positive_audio as gfp
import generate_true_positive_audio as gtp
import generate_false_negative_audio as gfn
import supporting_functions_for_audio_fetch as af

analyzer= Analyzer()

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