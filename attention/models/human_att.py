import json
import pandas as pd
from eyetrackpy.data_processor.models.eye_tracking_data_simple import (
    EyeTrackingDataUserSet,
)
import os


class HumanAttentionExtractor:
    def load_gaze_features(self, folder):
        datauserset = EyeTrackingDataUserSet()
        files = datauserset.search_word_coor_fixations_files(folder)
        fixations_trials = {}
        for trial, file in files.items():
            fixations_trials[trial] = datauserset._read_coor_trial(file)
        return fixations_trials


    def load_texts(self, path):
        fixations_trials = self.load_gaze_features(path)
        return {
            trial: list(fixations_trial.text)
            for trial, fixations_trial in fixations_trials.items()
        }

