import os

# Set the timeout to 5 seconds
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from models.human_att import HumanAttentionExtractor
from models.model_att import ModelAttentionExtractor
import argparse

path = "oasstetc_data/"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--reward",
        default=False,
    )
    args = parser.parse_args()

    models = {
        # -----------------------------------------------
        "llava-hf/llava-1.5-7b-hf": "VisionSeq",
    }

    users = range(1, 9)
    for model_name, model_type in models.items():
        for user_set in users:
            # Load the model
            model_name.replace("/", "_")

            folder_path_attention = (
                path
                + "attention_saliency/"
                + model_name.split("/")[1]
                + "/set_"
                + str(user_set)
                + "/"
            )
     
            folder_texts = path + "gaze_features_real" + "/set_" + str(user_set) + "/"
            texts_trials = HumanAttentionExtractor().load_texts(folder_texts)
            test_prompts = HumanAttentionExtractor.load_trial_prompts()
            att_extractor = ModelAttentionExtractor(model_name, model_type)
            word_level = True

            attention_trials = att_extractor.extract_attention(
                texts_trials, word_level=word_level
            )
           
            if word_level:
                att_extractor.save_attention_df(
                    attention_trials,
                    texts_trials=texts_trials,
                    path_folder=folder_path_attention,
                )
            else:
                att_extractor.save_attention_np(attention_trials, folder_path_attention)
