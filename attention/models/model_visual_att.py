# Load model directly
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
cwd = os.path.abspath(__file__)
sys.path.append(cwd)
import pandas as pd
from visual_processer import ImageProcessor
from model_att import ModelAttentionExtractor
from transformers import (
    BatchEncoding,
)
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
import numpy as np
import scipy.special
class ModelVisualAttentionExtractor():
    def __init__(self, model_name, model_type, folder_path_attention):
        self.model_name = model_name
        self.model_type = model_type
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            output_attentions=True,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )

         
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.folder_path_attention = folder_path_attention
        self.visual_processor = ImageProcessor()
        self.text_att = ModelAttentionExtractor(self.model, self.processor.tokenizer)
    

    def save_attention_df(self, *args, **kwargs):
        return self.text_att.save_attention_df(*args, **kwargs)

    def load_attention_df(self, *args, **kwargs):
        return self.text_att.load_attention_df(*args, **kwargs)

    def prepare_input(self, image_path, text):
        # Prepare image
        
        if isinstance(image_path, str):
            from PIL import Image
            image = Image.open(image_path)
        
        # Create messages for chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }
        ]
        # 1) Get the exact prompt STRING from the chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_tokenizer = self.processor.tokenizer(
            # text.split(' '),
            text,
            # is_split_into_words=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        words = [text[s:e] for s, e in inputs_tokenizer.offset_mapping[0]]
        return inputs, inputs_tokenizer
    

    def process_attention_text(self, input_ids, input_ids_tokenizer, text, list_word_original, text_token_idx, text_aggregated_attention):
        text_token = [int(input_ids['input_ids'][0][i]) for i in text_token_idx]
        first_token_ix_tokenizer = self.find_subsequence(input_ids_tokenizer['input_ids'][0], text_token)
        first_token_idx_procesor = self.find_subsequence(input_ids['input_ids'][0], text_token)
        list_idx_tokenizer = list(range(first_token_ix_tokenizer, first_token_ix_tokenizer + len(text_token)))
        list_idx_procesor = list(range(first_token_idx_procesor, first_token_idx_procesor + len(text_token)))
        corresponde_idx = dict(zip(list_idx_tokenizer, list_idx_procesor))
        text_aggregated_attention_tokenizer = []
        for i in list(range(len(input_ids_tokenizer['input_ids'][0]))):
            if i in corresponde_idx:
                text_aggregated_attention_tokenizer.append(text_aggregated_attention[corresponde_idx[i]])
            else:
                text_aggregated_attention_tokenizer.append(0)
        text_aggregated_attention = self.map_attention_from_words_to_words(
            list_words_first= list_word_original, 
            text= text,
            text_tokenized = input_ids_tokenizer,
            features_mapped_second_words = text_aggregated_attention_tokenizer)
        relative_attention_text = scipy.special.softmax(text_aggregated_attention)
        return relative_attention_text

    def extract_attention(self, texts_trials: dict, images_trials_paths: dict, attention_method ='rollout'):
        """
        Extract attention from visual LLM with both text and image inputs
        
        Args:
            texts_trials: dict with trial_id -> list of words
            images_trials: dict with trial_id -> image_path or PIL Image
            word_level: whether to return word-level or token-level attention
        """
        attention_trials_image = {}
        attention_trials_text = {}
        
        for trial, list_text in texts_trials.items():
            if '.' in str(trial):
                continue
                
            if int(trial) not in images_trials_paths:
                print(f"Warning: No image found for trial {trial}, skipping...")
                continue
                
            print("Processing trial", trial)
            
            # try:
            # Prepare text
            list_word_original = [str(x) for x in list_text]
            text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]
            inputs_id, inputs_id_tokenizer = self.prepare_input(images_trials_paths[int(trial)], text)
            

            # tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs_id['input_ids'][0])
            # print(f"Trial {trial} - Tokens: {tokens}")

            # decoded_text = self.processor.tokenizer.decode(inputs_id['input_ids'][0], skip_special_tokens=False)
            # print(f"Trial {trial} - Decoded text: {decoded_text}")
            # for i, token in enumerate(tokens):
            #     print(f"  Position {i}: '{token}' (ID: {inputs_id['input_ids'][0][i].item()})")

            info = self.visual_processor.map_single_image_tokens(inputs_id['input_ids'][0], self.processor.tokenizer, self.processor.image_processor, assume_patch_size=14)
            
            out_path = self.visual_processor.draw_all_patches_and_save(
                image_path=images_trials_paths[int(trial)],
                info=info,
                save_folder=self.folder_path_attention + "patch_plots/",
                image_name=f"patch_grid_{trial}.png",
                edgecolor="black",
                linewidth=0.6,
                annotate=False,   # set True if you want (r,c) labels
                dpi=200
            )

            text_token_idx, image_token_idx, special_token_idx = self.find_token_idx(inputs_id)   
            if attention_method =='rollout':
                print("Processing trial", trial, "with rollout")
                attention, confidences = self.get_attention_model_steps(self.model, inputs_id)
                
                attention_image_trial, attention_text_trial = self.process_attention_rollout(
                    attention = attention, 
                    input_ids = inputs_id, 
                    input_ids_tokenizer = inputs_id_tokenizer, 
                    text = text,
                    list_word_original = list_word_original, 
                    text_token_idx = text_token_idx,
                    image_token_idx = image_token_idx,
                    special_token_idx = special_token_idx,
                    info = info,
                    confidences = confidences,
                )
            else:
                attention = self.get_attention_model(self.model, inputs_id)
                
                attention_image_trial, attention_text_trial = self.process_attention(
                    attention = attention, 
                    input_ids = inputs_id, 
                    input_ids_tokenizer = inputs_id_tokenizer, 
                    text = text,
                    list_word_original = list_word_original, 
                    method = "mean_diagonal_below",
                    text_token_idx = text_token_idx,
                    image_token_idx = image_token_idx,
                    special_token_idx = special_token_idx,
                    info = info,
                )
                
            attention_trials_image[trial] = attention_image_trial
            attention_trials_text[trial] = attention_text_trial
                
        return attention_trials_image, attention_trials_text, info
    
    def save_attention_trials_image(self, images_trials_paths, attention_trials_image, info, path_save):
        for trial, attention_image_trial in attention_trials_image.items():
            for layer, attention_image in attention_image_trial.items():
                heat = attention_image["heatmap"]
                attention_image_layer = attention_image["attention"]
                #attention_image_layer hsa been already processed, we just need to convert it to a heatmap
                #the dimensions area already correct
                
                # out_path = self.visual_processor.save_attention_overlay(
                #     image_path=images_trials_paths[int(trial)],
                #     heat=heat,
                #     info=info,
                #     save_folder=self.folder_path_attention + "attention_overlay/",
                #     image_name=f"attention_overlay_{trial}_layer_{layer}.png",
                # )
                save_folder = path_save + "/trial_" + str(trial)
                os.makedirs(save_folder, exist_ok=True)
                save_attention_overlay_with_grid = self.visual_processor.save_attention_overlay_with_grid(
                    image_path=images_trials_paths[int(trial)],
                    heat=heat,
                    info=info,
                    save_folder=save_folder,
                    image_name=f"attention_overlay_{trial}_layer_{layer}.png",
                )
                # Save heatmap array
                np.save(os.path.join(save_folder, f"saliency_trial_{trial}_layer_{layer}.npy"), heat)
    @staticmethod
    def find_subsequence(seq, subseq):
            for i in range(len(seq) - len(subseq) + 1):
                if list(seq[i:i+len(subseq)]) == subseq:
                    return i
            return -1

    def find_image_token_positions(self, inputs):
        """Find the positions of <image> tokens in the sequence"""
        tok = self.processor.tokenizer
        
        # Find image token ID
        image_token_id = None
        if "<image>" in tok.get_vocab():
            image_token_id = tok.convert_tokens_to_ids("<image>")
        elif "additional_special_tokens" in tok.special_tokens_map:
            for s in tok.special_tokens_map["additional_special_tokens"]:
                if s == "<image>":
                    image_token_id = tok.convert_tokens_to_ids(s)
                    break
        
        if image_token_id is None:
            raise ValueError("Token <image> not found")
        
        input_ids = inputs["input_ids"][0]
        image_positions = (input_ids == image_token_id).nonzero(as_tuple=False).flatten()
        
        return image_positions.tolist(), len(image_positions)
    def find_token_idx(self, inputs):
        # Search subsequence (simple scan)
        
        special_token_idx = self.text_att.compute_special_token_idx(inputs['input_ids'][0], self.text_att.special_tokens_id)
        image_token_idx, _ = self.find_image_token_positions(inputs)
        text_token_idx = [x for x in range(len(inputs['input_ids'][0])) if x not in special_token_idx]
        special_token_idx = [x for x in special_token_idx if x not in image_token_idx]
        
        marker = "ASSISTANT:"
        marker_ids = self.processor.tokenizer(marker, add_special_tokens=False).input_ids
        assistant_pattern = self.find_subsequence(inputs['input_ids'][0], marker_ids)
        assistant_pattern_idx = list(range(assistant_pattern, assistant_pattern + len(marker_ids)))
        tokens_to_remove = [idx for idx in text_token_idx if idx < image_token_idx[-1] or idx in assistant_pattern_idx]
        text_token_idx = [idx for idx in text_token_idx if idx not in tokens_to_remove]
        special_token_idx.extend(tokens_to_remove)
        return text_token_idx, image_token_idx, special_token_idx



    def process_attention(
        self,
        attention,
        input_ids: BatchEncoding,
        input_ids_tokenizer: BatchEncoding,
        text: str = None,
        list_word_original: list = None,
        method="mean_diagonal_below",
        text_token_idx: list = None,
        image_token_idx: list = None,
        special_token_idx: list = None,
        info: dict = None,
    ):
         
        # image_token = [tokens[i] for i in image_token_idx]
        # text_token = [tokens[i] for i in text_token_idx]
        # special_token = [tokens[i] for i in special_token_idx]

        attention_layer_image = {}
        attention_layer_text = {}
        
        for layer in range(len(attention)):
            att_layer = attention[layer][0].cpu().detach().numpy()
            mean_attention = np.mean(att_layer, axis=0)
            # For each token, we sum over the attentions received from all other tokens.
            if method == "sum":
                aggregated_attention = np.sum(mean_attention, axis=0)
            elif method == "mean_diagonal_below":
                aggregated_attention = self.text_att.compute_mean_diagonalbewlow(
                    mean_attention
                )
            elif method == "last_token":
                # Use attention from just the last token
                aggregated_attention = mean_attention[-1, :]  # Last token's attention to all other tokens
            else:
                raise ValueError(f"Method {method} not supported")

            # aggregated_attention = [
            #     0 if i in special_token_idx else aggregated_attention[i]
            #     for i in range(len(aggregated_attention))
            # ]
            #-----image attention----------------------------------------------------
            # ---- choose queries and keys explicitly ----
            idx = np.asarray(image_token_idx)
            assert np.all(np.diff(idx) == 1), "image_token_idx must be contiguous & ascending"

            Q = np.asarray(text_token_idx, dtype=int)          # queries = TEXT ONLY
            K = np.asarray(image_token_idx, dtype=int)         # keys = IMAGE BLOCK (as in LLM seq)
            M_qk = mean_attention[np.ix_(Q, K)] 
            a_img = M_qk.sum(axis=0) / max(len(Q), 1)      
            # Optional: normalize only within the image block
            a_img = a_img / (a_img.sum() + 1e-12)
            relative_attention_image = a_img
            # image_aggregated_attention = [aggregated_attention[i] for i in image_token_idx]
            # relative_attention_image = scipy.special.softmax(image_aggregated_attention)
            # relative_attention_image = image_aggregated_attention
            heat = self.visual_processor.sequence_attention_to_patch_heatmap(
                    seq_attn=relative_attention_image,          # <- your 1D attention over keys
                    start=0,
                    gh=info["grid"][0],
                    gw=info["grid"][1],
                    has_cls=info["has_cls"]
                )
            #-----text attention-----
            text_aggregated_attention = [
                    aggregated_attention[i] if i in text_token_idx else 0
                    for i in range(len(aggregated_attention))
                ]
            relative_attention_text = self.process_attention_text(input_ids, input_ids_tokenizer, text, list_word_original, text_token_idx, text_aggregated_attention)
            #-----save attention-----
            attention_layer_image[layer] = {"attention": relative_attention_image, "heatmap": heat}
            attention_layer_text[layer] = relative_attention_text
        return attention_layer_image, attention_layer_text



    @staticmethod
    def map_attention_from_words_to_words(
        list_words_first: list,
        text: str,
        text_tokenized: BatchEncoding,
        features_mapped_second_words: list,
        mode="mean",
    ):
        list_words_second = [text[s:e].lower() for s, e in text_tokenized.offset_mapping[0]]
        # we map the original words (list_word_first) to the words in the text_tokenized
        mapped_words_idxs, mapped_words_str = TokenizerAligner().map_words(
            list_words_first, list_words_second
        )
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

    @staticmethod
    def get_attention_model_steps(model, inputs):
        # check if model has atribute device
        if not hasattr(model, "device"):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

        # Generate with attention
        with torch.no_grad():
            

            output = model.generate(
                **inputs,
                    do_sample=False,
                    max_new_tokens=200,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=True,
                    use_cache=True
                )

        input_length = inputs["input_ids"].size(1)
        generated_ids = output.sequences[0, input_length:]
        confidences = []
        for t, scores in enumerate(output.scores):
            probs = torch.softmax(scores[0], dim=-1)
            confidence = float(probs[generated_ids[t]].item())
            confidences.append(confidence)
        confidences
        return output.attentions, confidences


    def _compose_lastrow_rollout_np(self, last_rows, allowed_mask=None, eps=1e-9, alpha=0.25):
        """
        last_rows: list[num_layers] of 1D numpy arrays, length S (= L + t)
        allowed_mask: 1D np.float32 of length S with 1=allowed key, 0=forbidden
        alpha: residual strength; a_hat = (1-alpha)*a + alpha*e_last
        Returns:
            r: 1D np.array length S (rollout-composed distribution for the new token)
        """
        S = last_rows[0].shape[0]
        e_last = np.zeros(S, dtype=np.float32); e_last[-1] = 1.0

        if allowed_mask is None:
            allowed_mask = np.ones(S, dtype=np.float32)
        else:
            allowed_mask = allowed_mask.astype(np.float32, copy=False)
        allowed_mask[-1] = 1.0  # the new token (self) must always be allowed

        r = e_last.copy()
        for a in reversed(last_rows):
            a = a.astype(np.float32, copy=False)

            # mask out forbidden columns and renormalize ONLY over allowed keys
            a = a * allowed_mask
            denom = a.sum()
            if denom <= eps:
                # fallback: uniform over allowed keys
                denom = allowed_mask.sum() + eps
                a = allowed_mask / denom
            else:
                a = a / (denom + eps)

            # soft residual (alpha << 0.5 prevents drift to first tokens)
            a_hat = (1.0 - alpha) * a + alpha * e_last

            # efficient (r @ A_hat) when A_hat = I with last row replaced by a_hat
            r = r + r[-1] * (a_hat - e_last)

        return r


    def process_attention_rollout(
        self,
        attention,                       # tuple from generate(...).attentions, len N+1
        input_ids,
        input_ids_tokenizer,
        text: str = None,
        list_word_original: list = None,
        text_token_idx: list = None,
        image_token_idx: list = None,
        special_token_idx: list = None,
        info: dict = None,
        confidences: list = None,
        include_residual: bool = True,   # kept for API; alpha controls residual strength
        alpha: float = 0.25,             # <--- try 0.1–0.3; 0.0 = no residual, 0.5 ≈ (A+I)/2
        eps: float = 1e-9,
        skip_first_steps: int = 1        # often skip the first generated step (space/"The")
    ):
        assert info is not None and "grid" in info, "info['grid']=(gh,gw) is required"
        gh, gw = info["grid"]; n_patches = gh * gw

        prefill = attention[0]
        num_layers = len(prefill)
        L = prefill[0][0].shape[-1]
        N = len(attention) - 1
        T = L + N

        # ---- image keys K: trim to exactly gh*gw patches (drop leading specials; keep spatial) ----
        K = np.asarray(image_token_idx, dtype=int)
        if K.size > n_patches:
            K = K[-n_patches:]
        assert K.size == n_patches, f"Expected {n_patches} image keys, got {K.size}"

        # ---- build an ALLOWED mask over the full length: image patches + real text (no specials) ----
        allowed_full = np.zeros(T, dtype=np.float32)
        allowed_full[K] = 1.0
        if text_token_idx is not None:
            allowed_full[np.asarray(text_token_idx, dtype=int)] = 1.0
        if special_token_idx is not None:
            allowed_full[np.asarray(special_token_idx, dtype=int)] = 0.0  # exclude specials/prefix

        # accumulators
        acc_img = np.zeros(n_patches, dtype=np.float32)
        acc_text_full = np.zeros(T, dtype=np.float32)
        tot_w = 0.0

        # iterate generated steps
        for t in range(1 + skip_first_steps, N + 1):   # optionally skip very first steps
            step_layers = attention[t]                 # list[num_layers], each [1, H, 1, L+t]
            S = step_layers[0][0].shape[-1]

            # head-mean last row per layer -> numpy [S]
            last_rows = []
            for l in range(num_layers):
                row = step_layers[l][0].mean(dim=0)[0].detach().cpu().numpy()  # [S]
                last_rows.append(row)

            # per-step allowed mask (truncate to current length), keep last index allowed
            allowed_mask = allowed_full[:S].copy()
            allowed_mask[-1] = 1.0

            # rollout-composed row distribution r_t over S keys (masked + soft residual)
            r_t = self._compose_lastrow_rollout_np(last_rows, allowed_mask=allowed_mask,
                                                eps=eps, alpha=alpha)

            # optional step weight (confidence)
            w = float(confidences[t-1]) if (confidences is not None and t-1 < len(confidences)) else 1.0

            acc_img += w * r_t[K]          # K < S always (image keys are in the prompt)
            acc_text_full[:S] += w * r_t
            tot_w += w

        if tot_w == 0:
            tot_w = 1.0

        a_img = acc_img / tot_w
        a_img = a_img / (a_img.sum() + 1e-12)
        heat  = a_img.reshape(gh, gw)

        aggregated_attention_full = acc_text_full / tot_w
        text_aggregated_attention = [
            aggregated_attention_full[i] if (text_token_idx is not None and i in text_token_idx) else 0.0
            for i in range(T)
        ]
        relative_attention_text = self.process_attention_text(
            input_ids, input_ids_tokenizer, text, list_word_original,
            text_token_idx, text_aggregated_attention
        )

        attention_layer_image = {"gen_rollout": {"attention": a_img, "heatmap": heat}}
        attention_layer_text  = {"gen_rollout":  relative_attention_text}
        return attention_layer_image, attention_layer_text
