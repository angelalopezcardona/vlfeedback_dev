    
import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
from utils.data_loader import ETDataLoader
from utils.et_processer import ETProcessor
import numpy as np
import pandas as pd
if __name__ == "__main__":
    subjects = [1,2,3,4,5,6,7,8,9,10]
    # paths
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = cwd + "/data/raw/et"
    path_save = cwd + "/data/processed/et/agregated/"
    os.makedirs(path_save, exist_ok=True)
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    subject_fixations_word = ETDataLoader().load_subject_word_data(subjects=subjects, raw_data_path=raw_data_path)
    trials = [x  for x in list(subject_fixations_word[1].keys()) if '.' in x]
    ETProcessor().average_gaze_features_real_participants(subject_fixations_word, path_save, trials)

    

               