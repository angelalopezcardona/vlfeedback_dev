# Load model directly
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
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
from model_text_att import ModelTextAttentionExtractor

class ModelVisualAttentionExtractor():
    def __init__(self, model_name, model_type, folder_path_attention):
        self.model_name = model_name
        self.model_type = model_type
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            output_attentions=True,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.folder_path_attention = folder_path_attention
        self.visual_processor = ImageProcessor()
        self.text_att = ModelTextAttentionExtractor(self.model, self.processor.tokenizer)

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
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs

    def extract_attention(self, texts_trials: dict, images_trials_paths: dict, word_level: bool = True):
        """
        Extract attention from visual LLM with both text and image inputs
        
        Args:
            texts_trials: dict with trial_id -> list of words
            images_trials: dict with trial_id -> image_path or PIL Image
            word_level: whether to return word-level or token-level attention
        """
        attention_trials = {}
        
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
            list_word_original_lower = [x.lower() for x in list_word_original]
            inputs = self.prepare_input(images_trials_paths[int(trial)], text)
            

            tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            print(f"Trial {trial} - Tokens: {tokens}")

            decoded_text = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            print(f"Trial {trial} - Decoded text: {decoded_text}")
            for i, token in enumerate(tokens):
                print(f"  Position {i}: '{token}' (ID: {inputs['input_ids'][0][i].item()})")

            info = self.visual_processor.map_single_image_tokens(inputs['input_ids'][0], self.processor.tokenizer, self.processor.image_processor, assume_patch_size=14)
            
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
            
            attention = self.get_attention_model(self.model, inputs)
            special_token_idx = self.text_att.compute_special_token_idx(inputs['input_ids'][0], self.text_att.special_tokens_ids)
            image_token_idx = self.text_att.compute_special_token_idx(inputs['input_ids'][0], [self.processor.tokenizer.special_tokens_ids['image']])
            attention_image  = attention[0][0][info["start"]:info["start"] + info["length"]]
            attention_text = attention[0][0][info["start"] + info["length"]:]
            heat = sequence_attention_to_patch_heatmap(
                seq_attn=attention_image,          # <- your 1D attention over keys
                start=info["start"],
                gh=info["grid"][0],
                gw=info["grid"][1],
                has_cls=info["has_cls"]
            )
            out_path = self.visual_processor.save_attention_overlay(
                image_path=images_trials_paths[int(trial)],
                heat=heat,
                info=info,
                save_folder=self.folder_path_attention + "attention_overlay/",
                image_name=f"attention_overlay_{trial}.png",
            )
            attention_trials[trial] = [attention_image, attention_text, heat]
                
        return attention_trials
    
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