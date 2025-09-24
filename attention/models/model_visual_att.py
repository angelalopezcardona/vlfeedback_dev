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
from PIL import Image
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator
import cv2
import numpy as np
from visual_processer import ImageProcessor
from model_att import ModelAttentionExtractor
from transformers import (
    BatchEncoding,
)
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
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


    def backproject_patch_saliency(
        self,
        patch_scores: np.ndarray,         # (gh, gw) attention over patches
        original_image,                   # path or HxWx{1,3} np.ndarray (BGR ok)
        model_input_size: int = 336,      # CLIP/LLaVA crop size
        blur: bool = True,
        sigma_scale: float = 15,         # multiply the patch-size-based sigma
    ):
        """
        Map a patch-grid saliency/attention to the original image coordinates and (optionally) build an overlay.

        Steps (CLIP-style geometry):
        1) Upsample patch grid to SxS (NEAREST).
        2) Paste into a canvas of the resized image (short side = S) at the center-crop window.
        3) Resize that canvas back to the original HxW (NEAREST).
        4) (Optional) Gaussian blur in pixel space with σ ≈ patch size in original pixels.

        Returns
        -------
        saliency_raw : float32 HxW      # raw, aligned saliency (good for metrics)
        overlay      : uint8 HxWx3 or None
        """
        S = int(model_input_size)

        # --- load image ---
        if isinstance(original_image, str):
            img = cv2.imread(original_image, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {original_image}")
        else:
            img = original_image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        H, W = img.shape[:2]

        # --- 1) patch grid -> SxS (NEAREST) ---
        gh, gw = patch_scores.shape
        heat_S = cv2.resize(patch_scores.astype(np.float32), (S, S), interpolation=cv2.INTER_NEAREST)

        # --- 2) compute resized canvas size (short side -> S) and crop window ---
        scale = S / min(W, H)
        W_res = int(round(W * scale))
        H_res = int(round(H * scale))
        cx = max(0, (W_res - S) // 2)
        cy = max(0, (H_res - S) // 2)

        # paste SxS heatmap in the crop window on a canvas of the resized image
        canvas = np.zeros((H_res, W_res), dtype=np.float32)
        ex = min(cx + S, W_res)
        ey = min(cy + S, H_res)
        canvas[cy:ey, cx:ex] = heat_S[:ey - cy, :ex - cx]

        # --- 3) resize canvas back to original size (NEAREST) ---
        saliency_raw = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # --- 4) optional Gaussian smoothing in pixel space, σ tied to patch size in original pixels ---
        if blur:
            # patch size in canvas: S/gw (x), S/gh (y)
            # mapping to original: multiply by (W/W_res) and (H/H_res)
            patch_w_orig = (S / gw) * (W / max(W_res, 1))
            patch_h_orig = (S / gh) * (H / max(H_res, 1))
            # set σ so that FWHM ~ one patch (σ = patch/2.355). You can scale it with sigma_scale.
            sigmaX = max(0.1, (patch_w_orig / 2.355) * sigma_scale)
            sigmaY = max(0.1, (patch_h_orig / 2.355) * sigma_scale)
            saliency_raw = cv2.GaussianBlur(
                saliency_raw, (0, 0), sigmaX=sigmaX, sigmaY=sigmaY, borderType=cv2.BORDER_REPLICATE
            )

        return saliency_raw

    def prepare_input(self, image_path, text):

        if image_path is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.model.device, torch.float16)
        else:
            image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},                
                        {"type": "text", "text": text},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.model.device, torch.float16)

        inputs_tokenizer = self.processor.tokenizer(
            # text.split(' '),
            text,
            # is_split_into_words=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
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

    def extract_attention_only_text(self, texts_trials: dict, attention_method ='rollout'):
        """
        """
        attention_trials_text = {}
        
        for trial, list_text in texts_trials.items():
            if '.' not in str(trial):
                continue
                
            print("Processing response", trial)

            list_word_original = [str(x) for x in list_text]
            text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]
            inputs_id, inputs_id_tokenizer = self.prepare_input(None, text)
            
            text_token_idx, image_token_idx, special_token_idx = self.find_token_idx(inputs_id)   
            if attention_method =='rollout':
                print("Processing trial", trial, "with rollout")
                attention, confidences = self.get_attention_model_steps(self.model, inputs_id)
                
                _, attention_text_trial = self.process_attention_rollout(
                    attention = attention, 
                    input_ids = inputs_id, 
                    input_ids_tokenizer = inputs_id_tokenizer, 
                    text = text,
                    list_word_original = list_word_original, 
                    text_token_idx = text_token_idx,
                    image_token_idx = image_token_idx,
                    special_token_idx = special_token_idx,
                    info = None,  # No image info for text-only inputs
                    confidences = confidences,
                )
            else:
                attention = self.get_attention_model(self.model, inputs_id)
                
                _, attention_text_trial = self.process_attention(
                    attention = attention, 
                    input_ids = inputs_id, 
                    input_ids_tokenizer = inputs_id_tokenizer, 
                    text = text,
                    list_word_original = list_word_original, 
                    method = "mean_diagonal_below",
                    text_token_idx = text_token_idx,
                    image_token_idx = image_token_idx,
                    special_token_idx = special_token_idx,
                    info = None,  # No image info for text-only inputs
                )
                
           
            attention_trials_text[trial] = attention_text_trial
                
        return attention_trials_text

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
                
            for layer, attention_image in attention_image_trial.items():
                attention_image_trial[layer]["heatmap"] = self.backproject_patch_saliency(attention_image["heatmap"], images_trials_paths[int(trial)])
            attention_trials_image[trial] = attention_image_trial
            attention_trials_text[trial] = attention_text_trial
                
        return attention_trials_image, attention_trials_text, info
    
    def save_attention_trials_image(self, images_trials_paths, attention_trials_image, info, path_save):
        for trial, attention_image_trial in attention_trials_image.items():
            for layer, attention_image in attention_image_trial.items():
                heat = attention_image["heatmap"]
                attention_image_layer = attention_image["attention"]

                save_folder = path_save + "/trial_" + str(trial)
                os.makedirs(save_folder, exist_ok=True)
                np.save(os.path.join(save_folder, f"saliency_trial_{trial}_layer_{layer}.npy"), heat)
                overlay = SaliencyGenerator.create_overlay(image_path=images_trials_paths[int(trial)], saliency_map=heat)
                cv2.imwrite(os.path.join(save_folder, f"attention_overlay_{trial}_layer_{layer}.png"), overlay)
                # save_attention_overlay_with_grid = self.visual_processor.save_attention_overlay_with_grid(
                #     image_path=images_trials_paths[int(trial)],
                #     heat=heat,
                #     info=info,
                #     save_folder=save_folder,
                #     image_name=f"attention_overlay_{trial}_layer_{layer}.png",
                # )
                # Save heatmap array
    @staticmethod
    def find_subsequence(seq, subseq):
            for i in range(len(seq) - len(subseq) + 1):
                if list(seq[i:i+len(subseq)]) == subseq:
                    return i
            return -1

    # def find_image_token_positions(self, inputs):
    #     """Find the positions of <image> tokens in the sequence"""
    #     tok = self.processor.tokenizer
        
    #     # Find image token ID
    #     image_token_id = None
    #     if "<image>" in tok.get_vocab():
    #         image_token_id = tok.convert_tokens_to_ids("<image>")
    #     elif "additional_special_tokens" in tok.special_tokens_map:
    #         for s in tok.special_tokens_map["additional_special_tokens"]:
    #             if s == "<image>":
    #                 image_token_id = tok.convert_tokens_to_ids(s)
    #                 break
        
    #     if image_token_id is None:
    #         raise ValueError("Token <image> not found")
        
    #     input_ids = inputs["input_ids"][0]
    #     image_positions = (input_ids == image_token_id).nonzero(as_tuple=False).flatten()
        
    #     return image_positions.tolist(), len(image_positions)

    def find_token_idx(self, inputs):
        # Search subsequence (simple scan)
        
        special_token_idx = self.text_att.compute_special_token_idx(inputs['input_ids'][0], self.text_att.special_tokens_id)
        
        # Try to find image tokens, but handle case where they don't exist
        try:
            image_token_idx = self.text_att.compute_special_token_idx(inputs['input_ids'][0], [self.text_att.vocab[self.text_att.tokenizer.special_tokens_map['image_token']]])
        except (KeyError, AttributeError):
            # No image token found in vocabulary or special tokens map
            image_token_idx = []
        
        # image_token_idx, _ = self.find_image_token_positions(inputs)
        text_token_idx = [x for x in range(len(inputs['input_ids'][0])) if x not in special_token_idx]
        special_token_idx = [x for x in special_token_idx if x not in image_token_idx]
        
        marker = "ASSISTANT:"
        marker_ids = self.processor.tokenizer(marker, add_special_tokens=False).input_ids
        assistant_pattern = self.find_subsequence(inputs['input_ids'][0], marker_ids)
        assistant_pattern_idx = list(range(assistant_pattern, assistant_pattern + len(marker_ids)))
        if len(image_token_idx) > 0:
            tokens_to_remove = [idx for idx in text_token_idx if idx < image_token_idx[-1] or idx in assistant_pattern_idx]
        else:
            tokens_to_remove = [idx for idx in text_token_idx if idx in assistant_pattern_idx]
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
            if len(image_token_idx) > 0:
                idx = np.asarray(image_token_idx)
                assert np.all(np.diff(idx) == 1), "image_token_idx must be contiguous & ascending"

                Q = np.asarray(text_token_idx, dtype=int)          # queries = TEXT ONLY
                K = np.asarray(image_token_idx, dtype=int)         # keys = IMAGE BLOCK (as in LLM seq)
                M_qk = mean_attention[np.ix_(Q, K)] 
                a_img = M_qk.sum(axis=0) / max(len(Q), 1)      
                # Optional: normalize only within the image block
                a_img = a_img / (a_img.sum() + 1e-12)
                relative_attention_image = a_img

                # Handle grid info for heatmap reshaping
                if info is not None and "grid" in info:
                    heat = a_img.reshape(info["grid"][0], info["grid"][1])
                else:
                    # Default to 1x1 grid for text-only inputs
                    heat = a_img.reshape(1, 1)
            else:
                relative_attention_image = np.zeros(len(image_token_idx))
                heat = np.zeros((1, 1))
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
        # if not hasattr(model, "device"):
        #     input_ids = inputs["input_ids"]
        #     attention_mask = inputs["attention_mask"]
        # else:
        #     input_ids = inputs["input_ids"].to(model.device)
        #     attention_mask = inputs["attention_mask"].to(model.device)
            
        with torch.no_grad():
            output = model(
                **inputs,
                output_scores=True,
                output_attentions=True,
                )
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


    def  process_attention_rollout(
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
        # Handle text-only case (no image input)
        if info is None or "grid" not in info:
            # For text-only inputs, create dummy grid info
            gh, gw = 1, 1
            n_patches = 1
        else:
            gh, gw = info["grid"]
            n_patches = gh * gw

        prefill = attention[0]
        num_layers = len(prefill)
        L = prefill[0][0].shape[-1]
        N = len(attention) - 1
        T = L + N

        # ---- image keys K: handle case where there are no image tokens ----
        if image_token_idx is not None and len(image_token_idx) > 0:
            K = np.asarray(image_token_idx, dtype=int)
            if K.size > n_patches:
                K = K[-n_patches:]
            assert K.size == n_patches, f"Expected {n_patches} image keys, got {K.size}"
        else:
            # No image tokens - create empty array
            K = np.array([], dtype=int)

        # ---- build an ALLOWED mask over the full length: image patches + real text (no specials) ----
        allowed_full = np.zeros(T, dtype=np.float32)
        if len(K) > 0:
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

            # Only accumulate image attention if there are image tokens
            if len(K) > 0:
                acc_img += w * r_t[K]          # K < S always (image keys are in the prompt)
            acc_text_full[:S] += w * r_t
            tot_w += w

        if tot_w == 0:
            tot_w = 1.0

        # Handle image attention
        if len(K) > 0:
            a_img = acc_img / tot_w
            a_img = a_img / (a_img.sum() + 1e-12)
            heat = a_img.reshape(gh, gw)
        else:
            # No image tokens - return empty arrays
            a_img = np.zeros(n_patches, dtype=np.float32)
            heat = np.zeros((gh, gw), dtype=np.float32)

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
