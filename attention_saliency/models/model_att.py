import numpy as np
import scipy.special

import pandas as pd
import numpy as np

import os
import torch

from eyetrackpy.data_generator.models.fixations_aligner import FixationsAligner
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
from transformers import (
    BatchEncoding,
)

from models.model_loader import ModelLoaderFactory
from PIL import Image

class ModelAttentionExtractor:
    def __init__(self, model_name, model_type):
        self.model_type = model_type
        model_loader = ModelLoaderFactory().get_model_loader(model_type)
        self.model = model_loader.load_model(model_name)
        self.processor = model_loader.load_processor(model_name)
        #TODO: REVISE THIS
        # print(self.processor.chat_template)
        # vocab = self.processor.get_vocab()
        # self.special_tokens_id = [
        #     vocab[token] for _, token in self.tokenizer.special_tokens_map.items()
        # ]

    @staticmethod
    def map_attention_from_tokens_to_words_v2(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        mode="max",
    ):
        # funcon done for when
        # we compute the words list from the text_tokenized
        list_words_second = TokenizerAligner().tokens_to_words(text, text_tokenized)
        # we map the original words (list_word_first) to the words in the text_tokenized
        mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
            list_words_first, list_words_second
        )
        # we asign cero to the positions of the special tokens

        features_mapped_first_words = (
            TokenizerAligner().map_features_between_paired_list(
                features_mapped_second_words,
                mapped_words_idxs,
                list_words_first,
                mode=mode,
            )
        )
        return features_mapped_first_words

 

    @staticmethod
    def map_attention_from_words_to_words(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        mode="max",
    ):
        # we compute the words list from the text_tokenized
        list_words_second = TokenizerAligner().text_to_words(
            text, text_tokenized, text_tokenized.word_ids()
        )
        # we map the original words (list_word_first) to the words in the text_tokenized
        mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
            list_words_first, list_words_second
        )
        features_mapped_first_words = (
            TokenizerAligner().map_features_between_paired_list(
                features_mapped_second_words,
                mapped_words_idxs,
                list_words_first,
                mode="mean",
            )
        )
        return features_mapped_first_words

    @staticmethod
    def get_attention_model(model, inputs):
        # check if model has atribute device
        if not hasattr(model, "device"):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        return output.attentions

    # For the attention baseline, we fixed several experimental choices (see below) which might affect the results.

    def process_attention(
        self,
        attention,
        input_ids: BatchEncoding,
        text: str = None,
        list_word_original: list = None,
        word_level: bool = True,
        method="sum",
    ):
        attention_layer = {}
        special_token_idx = self.compute_special_token_idx(
            input_ids["input_ids"][0].tolist(), self.special_tokens_id
        )
        for layer in range(len(attention)):
            att_layer = attention[layer][0].cpu().detach().numpy()
            mean_attention = np.mean(att_layer, axis=0)
            # For each token, we sum over the attentions received from all other tokens.
            if method == "sum":
                aggregated_attention = np.sum(mean_attention, axis=0)
            else:
                if self.model_type == "BertBased":
                    aggregated_attention = np.mean(mean_attention, axis=0)
                else:
                    # we normalize the attention taking into account the decoder nature and the masking
                    # mean_attention_scaled_decoder = self.normalize_rows(mean_attention)
                    # we dont want to include the ceros because of masking
                    aggregated_attention = self.compute_mean_diagonalbewlow(
                        mean_attention
                    )

            if word_level is True:
                if list_word_original is None:
                    raise ValueError(
                        "list_word_original must be provided when word_level is True"
                    )
                if text is None:
                    raise ValueError("text must be provided when word_level is True")
                aggregated_attention = [
                    0 if i in special_token_idx else aggregated_attention[i]
                    for i in range(len(aggregated_attention))
                ]
                aggregated_attention_mapped_words = (
                    FixationsAligner().map_features_from_tokens_to_words(
                        aggregated_attention, input_ids, mode="sum"
                    )
                )
                if aggregated_attention_mapped_words is None:
                    aggregated_attention = self.map_attention_from_tokens_to_words_v2(
                        list_word_original,
                        text,
                        input_ids,
                        features_mapped_second_words=aggregated_attention,
                        mode="mean",
                    )
                else:
                    aggregated_attention = self.map_attention_from_words_to_words(
                        list_word_original,
                        text,
                        input_ids,
                        aggregated_attention_mapped_words,
                        mode="mean",
                    )

            else:
                # to compute attention per token we remove special tokens
                aggregated_attention = np.delete(
                    aggregated_attention, special_token_idx
                )
            relative_attention = scipy.special.softmax(aggregated_attention)
            # relative_attention = aggregated_attention
            attention_layer[layer] = relative_attention
        return attention_layer

    def extract_attention(self, texts_trials: dict, word_level: bool = True):
        attention_trials = {}
        for trial, list_text in texts_trials.items():
            print("trial", trial)
            list_word_original = [str(x) for x in list_text]
            text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]
            input_ids = self.process_prompt(self.processor, text, image)
            attention = self.get_attention_model(self.model, input_ids)
            try:
                attention_trials[trial] = self.process_attention(
                    attention,
                    input_ids,
                    text=text,
                    list_word_original=list_word_original,
                    word_level=word_level,
                )
            except Exception as e:
                print(trial, "error:", e)
        return attention_trials



    @staticmethod
    def normalize_rows(matrix):
        cols_normalized = []
        for row in range(matrix.shape[0]):
            elements = matrix[row,]
            cols_normalized.append(elements / (1 / (row + 1)))
        return np.array(cols_normalized)

    @staticmethod
    def compute_mean_diagonalbewlow(matrix):
        means = []
        for col in range(matrix.shape[1]):
            # Select elements in the current column from the diagonal and below
            valid_elements = matrix[col:, col]  # Start from the current row downwards
            means.append(
                np.mean(valid_elements)
            )  # Calculate the mean of these elements
        return np.array(means)


    @staticmethod
    def compute_special_token_idx(tokens_list, special_tokens_ids):
        special_token_idx = []
        for i, token in enumerate(tokens_list):
            if token in special_tokens_ids:
                special_token_idx.append(i)
        return special_token_idx

    @staticmethod
    def process_prompt(processor, text, image):
        
model_names =[
            #   "llava-hf/llava-1.5-7b-hf", 
              "meta-llama/Llama-3.2-11B-Vision-Instruct", 
            #   "google/paligemma-3b-mix-224"
            ]
trial_name = '5_baseline.jpg'
image = Image.open('data' +   "/" + trial_name)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                    {"type": "text", "text": text}
                ]
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        return  inputs

    @staticmethod
    def save_attention_np(attention_trials, path_folder):
        for trial, attention_layer in attention_trials.items():
            path_folder_trial = path_folder + "trial_" + str(trial)
            if not os.path.exists(path_folder_trial):
                os.makedirs(path_folder_trial)
            for layer, attention in attention_layer.items():
                np.save(path_folder_trial + "/layer_" + str(layer) + ".npy", attention)

    @staticmethod
    def save_attention_df(attention_trials, texts_trials, path_folder):
        for trial, attention_layer in attention_trials.items():
            path_folder_trial = path_folder + "trial_" + str(trial)
            if not os.path.exists(path_folder_trial):
                os.makedirs(path_folder_trial)
            trial_text = texts_trials[trial]
            for layer, attention in attention_layer.items():
                pd.DataFrame({"text": trial_text, "attention": attention}).to_csv(
                    path_folder_trial + "/layer_" + str(layer) + ".csv",
                    sep=";",
                    index=False,
                )

    @staticmethod
    def load_attention_np(path_folder):
        attention_trials = {}
        for trial in os.listdir(path_folder):
            attention_layer = {}
            for layer in os.listdir(path_folder + "/" + trial):
                attention = np.load(path_folder + "/" + trial + "/" + layer)
                attention_layer[int(layer.split("_")[1].split(".")[0])] = attention
            attention_trials[float(trial.split("_")[1])] = attention_layer
        return attention_trials

    @staticmethod
    def load_attention_df(path_folder):
        attention_trials = {}
        for trial in os.listdir(path_folder):
            attention_layer = {}
            for layer in os.listdir(path_folder + "/" + trial):
                attention = pd.read_csv(
                    path_folder + "/" + trial + "/" + layer, sep=";"
                )
                attention_layer[int(layer.split("_")[1].split(".")[0])] = attention
            attention_trials[float(trial.split("_")[1])] = attention_layer
        return attention_trials
