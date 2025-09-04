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

from model_att import ModelAttentionExtractor
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

class ModelVisualAttentionExtractor():
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            output_attentions=True,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
                
            if trial not in images_trials_paths:
                print(f"Warning: No image found for trial {trial}, skipping...")
                continue
                
            print("Processing trial", trial)
            
            try:
                # Prepare text
                list_word_original = [str(x) for x in list_text]
                text = " ".join(list_word_original)
                list_word_original_lower = [x.lower() for x in list_word_original]
                
                # Prepare image
                image = images_trials_paths[trial]
                if isinstance(image, str):
                    from PIL import Image
                    image = Image.open(image)
                
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
                
                # Process with the processor
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
                
                # Get attention by running forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                    attention = outputs.attentions
                
                # Extract attention for the input tokens (not generated tokens)
                input_length = inputs["input_ids"].shape[-1]
                attention_trials[trial] = self.process_visual_attention(
                    attention,
                    inputs["input_ids"],
                    text=text,
                    list_word_original=list_word_original_lower,
                    word_level=word_level,
                    processor=self.processor
                )
                
            except Exception as e:
                print(f"Error processing trial {trial}: {e}")
                continue
                
        return attention_trials
    
    def process_visual_attention(self, attention, input_ids, text, list_word_original, word_level=True, processor=None):
        """
        Process attention weights for visual LLM, mapping tokens back to words
        """
        import torch
        
        # Get the last layer attention (or average across layers)
        if isinstance(attention, tuple):
            # Average across all layers
            attention = torch.stack(attention).mean(dim=0)
        
        # attention shape: [batch_size, num_heads, seq_len, seq_len]
        attention = attention.squeeze(0)  # Remove batch dimension
        
        # Decode tokens to understand what each token represents
        tokens = processor.decode(input_ids[0], skip_special_tokens=False)
        token_list = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        print(f"Text: {text}")
        print(f"Tokens: {token_list}")
        
        if word_level:
            # Map tokens to words
            word_attention = self.map_tokens_to_words(
                attention, token_list, list_word_original, processor
            )
            return word_attention
        else:
            # Return token-level attention
            return {
                'attention': attention.cpu().numpy(),
                'tokens': token_list,
                'text': text
            }
    
    def map_tokens_to_words(self, attention, token_list, word_list, processor):
        """
        Map token-level attention to word-level attention
        """
        import torch
        import numpy as np
        
        # This is a simplified mapping - you might need to adjust based on your specific tokenizer
        word_attention = []
        
        # Find which tokens correspond to which words
        # This is model-specific and might need adjustment
        current_word_idx = 0
        token_to_word = []
        
        for i, token in enumerate(token_list):
            if token.startswith('▁') or token.startswith('##'):
                # New word token
                if current_word_idx < len(word_list):
                    token_to_word.append(current_word_idx)
                    current_word_idx += 1
                else:
                    token_to_word.append(current_word_idx - 1)
            else:
                # Continuation of current word
                if current_word_idx > 0:
                    token_to_word.append(current_word_idx - 1)
                else:
                    token_to_word.append(0)
        
        # Aggregate attention by word
        num_words = len(word_list)
        word_attention_matrix = np.zeros((num_words, num_words))
        
        for i in range(len(token_to_word)):
            for j in range(len(token_to_word)):
                word_i = token_to_word[i]
                word_j = token_to_word[j]
                if word_i < num_words and word_j < num_words:
                    # Average attention across heads
                    word_attention_matrix[word_i, word_j] += attention[:, i, j].mean().item()
        
        return {
            'attention': word_attention_matrix,
            'words': word_list,
            'tokens': token_list,
            'token_to_word_mapping': token_to_word
        }
    def extract_visual_attention(self, texts_trials: dict, images_trials_paths: dict, word_level: bool = True):
        """
        Extract attention from visual tokens and map back to image regions
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import cv2
        
        attention_trials = {}
        
        for trial, list_text in texts_trials.items():
            if '.' in str(trial):
                continue
                
            if trial not in images_trials_paths:
                print(f"Warning: No image found for trial {trial}, skipping...")
                continue
                
            print("Processing visual attention for trial", trial)
            
            try:
                # Prepare text and image
                list_word_original = [str(x) for x in list_text]
                text = " ".join(list_word_original)
                
                image_path = images_trials_paths[trial]
                image = Image.open(image_path) if isinstance(image_path, str) else image_path
                
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
                
                # Process with the processor
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
                
                # Get attention by running forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                    attention = outputs.attentions
                
                # Extract visual attention
                visual_attention = self.process_visual_attention_maps(
                    attention, inputs, image, text, processor=self.processor
                )
                
                attention_trials[trial] = visual_attention
                
            except Exception as e:
                print(f"Error processing trial {trial}: {e}")
                continue
                
        return attention_trials
    
    def process_visual_attention_maps(self, attention, inputs, image, text, processor):
        """
        Process attention to create visual attention maps
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        
        # Get the last layer attention (or average across layers)
        if isinstance(attention, tuple):
            attention = torch.stack(attention).mean(dim=0)
        
        # attention shape: [batch_size, num_heads, seq_len, seq_len]
        attention = attention.squeeze(0)  # Remove batch dimension
        
        # Get token information
        token_list = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Find visual tokens (usually at the beginning of the sequence)
        # For LLaVA, visual tokens are typically the first tokens after special tokens
        visual_token_start = self.find_visual_token_start(token_list)
        text_token_start = self.find_text_token_start(token_list, visual_token_start)
        
        print(f"Visual tokens start at: {visual_token_start}")
        print(f"Text tokens start at: {text_token_start}")
        print(f"Total tokens: {len(token_list)}")
        
        # Extract attention from text tokens to visual tokens
        if text_token_start < len(token_list):
            # Get attention from text tokens to visual tokens
            text_to_visual_attention = attention[:, text_token_start:, visual_token_start:text_token_start]
            
            # Average across heads and text tokens
            visual_attention_weights = text_to_visual_attention.mean(dim=(0, 1))  # [num_visual_tokens]
            
            # Create attention map
            attention_map = self.create_visual_attention_map(
                visual_attention_weights, image, visual_token_start, text_token_start
            )
            
            return {
                'attention_map': attention_map,
                'visual_attention_weights': visual_attention_weights.cpu().numpy(),
                'text_tokens': token_list[text_token_start:],
                'visual_tokens': token_list[visual_token_start:text_token_start],
                'image': image,
                'text': text
            }
        else:
            print("No text tokens found")
            return None
    
    def find_visual_token_start(self, token_list):
        """
        Find where visual tokens start in the token sequence
        """
        # Look for common visual token patterns
        for i, token in enumerate(token_list):
            if token in ['<image>', '<|image_pad|>', '<|start_of_image|>']:
                return i + 1
            # For some models, visual tokens might start after special tokens
            if token.startswith('<') and 'image' in token.lower():
                return i + 1
        return 0  # Default to start if not found
    
    def find_text_token_start(self, token_list, visual_start):
        """
        Find where text tokens start
        """
        # Look for text start markers
        for i in range(visual_start, len(token_list)):
            if token_list[i] in ['<|start_of_text|>', '<s>', '<|im_start|>']:
                return i + 1
        return visual_start + 256  # Default assumption for visual token length
    
    def create_visual_attention_map(self, attention_weights, image, visual_start, text_start):
        """
        Create attention map from visual token attention weights
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        from PIL import Image
        import cv2
        
        # Convert image to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Get image dimensions
        h, w = image_np.shape[:2]
        
        # Calculate number of visual tokens
        num_visual_tokens = text_start - visual_start
        
        # For LLaVA, visual tokens typically represent patches
        # Calculate patch size (this is model-specific)
        patch_size = self.calculate_patch_size(h, w, num_visual_tokens)
        
        # Reshape attention weights to spatial dimensions
        if patch_size is not None:
            patch_h, patch_w = patch_size
            attention_map = attention_weights.reshape(patch_h, patch_w)
            
            # Resize to original image size
            attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            # Fallback: create a simple attention map
            attention_map = np.ones((h, w)) * attention_weights.mean().item()
        
        return attention_map
    
    def calculate_patch_size(self, h, w, num_tokens):
        """
        Calculate patch size for visual tokens
        """
        # For LLaVA, this is typically 14x14 or 16x16 patches
        # Try common patch sizes
        for patch_size in [14, 16, 24, 32]:
            if patch_size * patch_size == num_tokens:
                return (patch_size, patch_size)
        
        # If no perfect square, try to find close dimensions
        import math
        sqrt_tokens = int(math.sqrt(num_tokens))
        if sqrt_tokens * sqrt_tokens == num_tokens:
            return (sqrt_tokens, sqrt_tokens)
        
        return None
    
    def plot_visual_attention(self, attention_data, save_path=None):
        """
        Plot attention map overlaid on the original image
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(attention_data['image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        attention_map = attention_data['attention_map']
        im1 = axes[1].imshow(attention_map, cmap='hot', alpha=0.8)
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(attention_data['image'])
        axes[2].imshow(attention_map, cmap='hot', alpha=0.6)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
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