from eyetrackpy.data_processor.models.eye_tracking_data_simple import (
    EyeTrackingDataUserSet,
)
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
import pandas as pd
import json
import os

def check_same_shape(dfs_dict):
    # Get the shape of the first DataFrame
    first_shape = list(dfs_dict.values())[0].shape
    # Compare the shape of all DataFrames to the first one
    return all(df.shape == first_shape for df in dfs_dict.values())


def save_to_json(data, filename="data.json"):
    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(data, f)


class ETProcessor:
    @staticmethod
    def arrange_dimensions(wor_cor, columns_to_calculate):
        shapes = [df.shape[0] for df in wor_cor.values()]
        # Get the index of the least common element
        index_min_shape = [i for i, x in enumerate(shapes) if x == min(shapes)]
        index_max_shape = [i for i, x in enumerate(shapes) if x == max(shapes)]
        # Get the DataFrame with the least common shape
        correct_df = list(wor_cor.values())[index_min_shape[0]]
        list_words_first = correct_df["text"].values.tolist()
        for index in index_max_shape:
            df = list(wor_cor.values())[index]
            list_words_second = df["text"].values.tolist()
            mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
                list_words_first, list_words_second
            )
            new_df = {"text": list_words_first}
            for column in list(columns_to_calculate.keys()):
                new_df[column] = TokenizerAligner().map_features_between_paired_list(
                    df[column].values.tolist(),
                    mapped_words_idxs,
                    list_words_first,
                    mode="mean",
                )
            new_df = pd.DataFrame(new_df)
            new_df.index = correct_df.index
            # create normalized columns
            for col, col_n in columns_to_calculate.items():
                new_df[col_n] = new_df[col] / new_df[col].sum()

            wor_cor[list(wor_cor.keys())[index]] = new_df

        return wor_cor
    

    def average_gaze_features_real_participants(self, fixations_participants, path_save, trials):
            
            # compute_texts()
            datauserset = EyeTrackingDataUserSet()


            if not os.path.exists(path_save):
                os.makedirs(path_save)
            # -----------------------------------------------------------------

            columns_to_calculate = {
                "fix_duration": "fix_duration_n",
                "first_fix_duration": "first_fix_duration_n",
                "fix_number": "fix_number_n",
            }
            for trial in trials:
                wor_cor = {}
                for participant, data_participant_all in fixations_participants.items():
                    if trial not in data_participant_all:
                        print("Trial not found for participant", participant, trial)
                        continue
                    data_participant = data_participant_all[trial]
                    # if not isinstance(data_participant, pd.DataFrame):
                    #     continue
                    # missing_col = False
                    # for column in list(columns_to_calculate.keys()):
                    #     if column not in data_participant.columns:
                    #         missing_col = True
                    #         break
                    # if missing_col is True:
                    #     continue
                    for col, col_n in columns_to_calculate.items():
                        data_participant[col_n] = (
                            data_participant[col] / data_participant[col].sum()
                        )
                    wor_cor[participant] = data_participant
                    wor_cor[participant] = wor_cor[participant].set_index("number")

                if wor_cor == {}:
                    continue
                # Stack all dataframes on top of each other and then group by the index
                same_dimension = check_same_shape(wor_cor)
                if not same_dimension:
                    print("Different dimensions for reial", trial)
                    print([x for x in wor_cor.keys()])
                    print([df.shape[0] for df in wor_cor.values()])
                    wor_cor = self.arrange_dimensions(
                        wor_cor, columns_to_calculate
                    )
                    print("After correcting", trial)
                    print([x for x in wor_cor.keys()])
                    print([df.shape[0] for df in wor_cor.values()])
                # change index to text in all dataframes
                columns_to_calculate_list = list(columns_to_calculate.keys())
                columns_to_calculate_list.extend(list(columns_to_calculate.values()))
                combined_df = pd.concat(
                    [df[columns_to_calculate_list] for df in wor_cor.values()]
                )
                # Now calculate the mean for each column, grouping by index to match rows across dataframes
                wor_cor_all = combined_df.groupby(combined_df.index).mean()
                wor_cor_all["text"] = list(wor_cor.values())[0]["text"]
                datauserset.save_words_fixations_trial(
                    path_save, trial, wor_cor_all
                )
                