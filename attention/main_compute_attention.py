import os

# Set the timeout to 5 seconds
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from utils.data_loader import ETDataLoader
from models.model_att import ModelAttentionExtractor
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
        "google-bert/bert-base-uncased": "BertBased",
        "google-bert/bert-large-uncased": "BertBased",
        "google-bert/bert-base-cased": "BertBased",
        "FacebookAI/roberta-base": "BertBased",
        "FacebookAI/roberta-large": "BertBased",
        # -----------------------------------------------
        "mistralai/Mistral-7B-v0.1":"causalLM",
        "mistralai/Mistral-7B-Instruct-v0.1": "causalLM",
        "meta-llama/Llama-2-7b-hf": "causalLM",
        "meta-llama/Llama-2-7b-chat-hf": "causalLM",
        # -----------------------------------------------
        "meta-llama/Meta-Llama-3-8B": "causalLM",
        "meta-llama/Meta-Llama-3-8B-Instruct": "causalLM",
        "meta-llama/Llama-3.1-8B": "causalLM",
        "meta-llama/Llama-3.1-8B-Instruct": "causalLM",
        #-----------------------------------------------
        "microsoft/phi-1_5": "causalLM",
        "openbmb/UltraRM-13b": "ultraRM",
        "openbmb/Eurus-RM-7b": "eurus",
        "nicolinho/QRM-Llama3.1-8B": "QRLlama",
    }

    users = range(1, 11)
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et"
    save_path= cwd + '/results/et/'

    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    for model_name, model_type in models.items():
        for user_set in users:
            # Load the model
            model_name.replace("/", "_")
            if reward is False:
                folder_path_attention = (
                    save_path
                    + "attention/"
                    + model_name.split("/")[1]
                    + "/set_"
                    + str(user_set)
                    + "/"
                )
            else:
                folder_path_attention = (
                    save_path
                    + "attention_reward/"
                    + model_name.split("/")[1]
                    + "/set_"
                    + str(user_set)
                    + "/"
                )
            
            att_extractor = ModelAttentionExtractor(model_name, model_type)
            word_level = True
            if reward is False:
                attention_trials = att_extractor.extract_attention(
                    responses, word_level=word_level
                )
            else:
                attention_trials = att_extractor.extract_attention_reward(
                    responses, texts_promps=prompts
                )
            if word_level:
                att_extractor.save_attention_df(
                    attention_trials,
                    texts_trials=responses,
                    path_folder=folder_path_attention,
                )
            else:
                att_extractor.save_attention_np(attention_trials, folder_path_attention)
