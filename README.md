# man vs BirdNet : Comprehensive guide to using the code
### Introduction
This repo/ set of python files is for testing BirdNet model accuracy on audio files with ground truth data. It is also to answer questions like what parameters (like species identity, confidence thresholds) enhance/worsen the accuracy of the algorithm.

### General instructions
1. Audio files that were used by me can be found in < insert link >. These have to be copied into the repo directory.
2. Depending on use case you can either run "run_codes.py" for single time interval and conf or use "run_codes_variable_conf.py" for a range of conf values. Please change the conf values to what is suitable for you.
3. Make sure that parent directory and directory of sound files is modified in your code. Change this in "run_codes.py" and "run_codes_variable_conf.py" as fits your need.
