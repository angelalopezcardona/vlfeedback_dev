import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
cwd = os.path.abspath(__file__)
sys.path.append(cwd)
# Set the timeout to 5 seconds
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from utils.data_loader import ETDataLoader
from attention.models.model_visual_att import ModelVisualAttentionExtractor
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--reward",
        default=False,
    )
    args = parser.parse_args()
    reward = str(args.reward).lower() == "true"
    print(f"Reward variable:{reward, type(reward)}")

    models = {
        # -----------------------------------------------
        "llava-hf/llava-1.5-7b-hf": "causalLM",
    }

    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et"
    save_path= cwd + '/results/et/'

    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    path_images = raw_data_path + "/images/"
    images_trials_paths = {trial: path_images + "img_prompt_" + str(trial) + ".jpg" for trial, image in prompts.items()}
    words = ETDataLoader().load_texts_responses(raw_data_path=raw_data_path)
    responses_words  ={}
    prompts_words = {}
    for trial, list_text in words.items():
        if not '.' in str(trial):
            prompts_words[trial] = list_text
        else:
            responses_words[trial] = list_text

    for model_name, model_type in models.items():
        # Load the model
        model_name.replace("/", "_")

        folder_path_attention = (
            save_path
            + "attention/"
            + model_name.split("/")[1]
            + "/"
        )
       
        
        att_extractor = ModelVisualAttentionExtractor(model_name, model_type)
        word_level = True

        attention_trials = att_extractor.extract_attention(
            responses_words, word_level=word_level, images_trials_paths=images_trials_paths
        )
    
        att_extractor.save_attention_df(
            attention_trials,
            texts_trials=responses_words,
            path_folder=folder_path_attention,
        )
        
        att_extractor.save_attention_np(attention_trials, folder_path_attention)
        
        