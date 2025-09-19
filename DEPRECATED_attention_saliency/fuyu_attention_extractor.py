import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import FuyuProcessor, FuyuForCausalLM
import cv2
import os
from scipy.ndimage import zoom
import torch.nn.functional as F
import math

class AttentionAnalyzer:
    """Class to handle attention analysis for Fuyu model"""
    
    def __init__(self, model_id="adept/fuyu-8b", device_id=0):
        """Initialize the attention analyzer with model and processor"""
        self.model_id = model_id
        self.device_id = device_id
        self.model = None
        self.processor = None
        self.attention = None
        self.len_input_ids = None
        self.image_positions = None
        self.image_tokens = None
        self.grid_size_H = None
        self.grid_size_W = None
        self._load_model()
    
    def _load_model(self):
        """Load the Fuyu model and processor"""
        self.model = FuyuForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            device_map=f"cuda:{self.device_id}",
            attn_implementation="eager"
        )
        self.processor = FuyuProcessor.from_pretrained(self.model_id)
    
    def prepare_inputs(self, image_path, prompt_text):
        """Prepare inputs for the model"""
        raw_image = Image.open(image_path).convert('RGB')
        H, W = raw_image.size
        self.grid_size_H = W
        self.grid_size_W = H
        inputs = self.processor(text=prompt_text, images=raw_image, return_tensors="pt").to(f"cuda:{self.device_id}")
        
        return inputs, raw_image
    
    def find_image_token_positions(self, inputs, prompt_text=None):
        """Find the positions of |SPEAKER| tokens in the sequence"""
        tok = self.processor.tokenizer
        
        # Find |SPEAKER| token ID
        speaker_token_id = None
        if "|SPEAKER|" in tok.get_vocab():
            speaker_token_id = tok.convert_tokens_to_ids("|SPEAKER|")
        elif "additional_special_tokens" in tok.special_tokens_map:
            for s in tok.special_tokens_map["additional_special_tokens"]:
                if s == "|SPEAKER|":
                    speaker_token_id = tok.convert_tokens_to_ids(s)
                    break
        
        if speaker_token_id is None:
            raise ValueError("Token |SPEAKER| not found")
        
        input_ids = inputs["input_ids"][0]
        image_positions = (input_ids == speaker_token_id).nonzero(as_tuple=False).flatten()
        
        return image_positions.tolist(), len(image_positions)
    
    def compute_attention_heatmap(self, inputs, prompt_text=None):
        """Compute attention heatmap efficiently with a single forward pass"""
        
        # Find image token positions
        image_positions, num_image_tokens = self.find_image_token_positions(inputs, prompt_text)
        self.image_positions = image_positions
        self.image_tokens = num_image_tokens
        
        # Generate with attention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=200,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
                use_cache=True
            )
        
        # Extract generated tokens and confidence
        input_length = inputs["input_ids"].size(1)
        self.len_input_ids = input_length
        generated_ids = outputs.sequences[0, input_length:]
        num_generated = len(generated_ids)
        
        # Calculate confidence for each generated token
        confidences = []
        for t, scores in enumerate(outputs.scores):
            probs = torch.softmax(scores[0], dim=-1)
            confidence = float(probs[generated_ids[t]].item())
            confidences.append(confidence)
        self.confidences = confidences
        
        # Process attention efficiently
        device = inputs["input_ids"].device
        accumulated_attention = torch.zeros(num_image_tokens, dtype=torch.float32, device=device)
        total_weight = 0.0
        
        lambda_d = 0.3  # Depth prior
        self.attention = outputs.attentions
        
        for t, step_attentions in enumerate(outputs.attentions):
            if t >= len(confidences):
                break
                
            token_confidence = confidences[t]
            step_attention_sum = torch.zeros(num_image_tokens, dtype=torch.float32, device=device)
            step_weight = 0.0
            
            # Process all layers of this step
            for layer_idx, layer_attention in enumerate(step_attentions):
                layer_weight = np.exp(lambda_d * (layer_idx + 1))
                
                # Get the last token (just generated) attending to image tokens
                last_token_attention = layer_attention[0, :, -1, :]  # (heads, total_seq_len)
                
                # Extract attention to image tokens
                img_attention = last_token_attention[:, image_positions]  # (heads, num_image_tokens)
                
                # Average over heads
                avg_attention = img_attention.mean(dim=0)  # (num_image_tokens,)
                
                step_attention_sum += layer_weight * avg_attention
                step_weight += layer_weight
            
            # Normalize by layer weight and accumulate weighted by confidence
            if step_weight > 0:
                normalized_step_attention = step_attention_sum / step_weight
                accumulated_attention += token_confidence * normalized_step_attention
                total_weight += token_confidence
        
        # Final normalization and reshape
        if total_weight > 0:
            final_attention = accumulated_attention / total_weight
        else:
            final_attention = accumulated_attention
        
        # Move to CPU and reshape using the calculated grid dimensions
        heatmap = final_attention.cpu().numpy().reshape(self.grid_size_H, self.grid_size_W)
        
        return heatmap, confidences, num_generated
    
    def create_attention_overlay_fuyu(
        self,
        heatmap,                      # 1D (Npatch) or 2D (rows x cols), ONLY patches (no newline)
        original_pil_image,           # PIL.Image
        patch_size=30,                # Fuyu uses 30x30 patches
        alpha=0.6,
        output_path=None
    ):
        """
        Create attention overlay for Fuyu model with proper patch-based visualization.
        
        This function takes a 2D attention heatmap from Fuyu's attention rollout and creates
        a visual overlay on the original image. Fuyu processes images in 30x30 pixel patches,
        so this function:
        
        1. Takes the attention values for each patch (17x13 grid for typical images)
        2. Maps each patch's attention value to the corresponding 30x30 pixel region
        3. Applies Gaussian blur for smoother visualization
        4. Creates a color-coded overlay using the JET colormap
        5. Blends the overlay with the original image
        
        Args:
            heatmap: 2D numpy array or torch tensor with attention values per patch
            original_pil_image: Original PIL image to overlay attention on
            patch_size: Size of each patch in pixels (30 for Fuyu)
            alpha: Transparency of the overlay (0.0 = transparent, 1.0 = opaque)
            output_path: Optional path to save the overlay image and numpy array
            
        Returns:
            overlay_pil: PIL image with attention overlay
            sal_uint8: Normalized saliency map as uint8 numpy array
        """

        # --- 0) Convert PIL image to OpenCV format
        orig_bgr = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
        H, W = orig_bgr.shape[:2]

        # --- 1) Convert heatmap to numpy float32
        if isinstance(heatmap, torch.Tensor):
            h = heatmap.detach().float().cpu().numpy()
        else:
            h = np.asarray(heatmap, dtype=np.float32)

        # --- 2) Determine rows/cols from heatmap shape
        import math
        rows, cols = heatmap.shape
        if h.ndim == 1:
            assert h.size == rows * cols, \
                f"Heatmap size={h.size} doesn't match rows*cols={rows*cols}. "\
                "Check that NEWLINE tokens are excluded and size is native."
            h2d = h.reshape(rows, cols)
        elif h.ndim == 2:
            assert h.shape == (rows, cols), \
                f"Expected 2D heatmap {rows}x{cols}, but got {h.shape}. "\
                "Check the patching (ceil(H/30), ceil(W/30))."
            h2d = h
        else:
            raise ValueError("heatmap must be 1D (Npatch) or 2D (rows x cols)")

        # --- 3) Build the map at ORIGINAL resolution by filling patch blocks
        map_px = np.zeros((H, W), dtype=np.float32)
        for r in range(rows):
            y1 = r * patch_size
            y2 = min(y1 + patch_size, H)  # last row can be shorter
            for c in range(cols):
                x1 = c * patch_size
                x2 = min(x1 + patch_size, W)  # last column can be narrower
                val = h2d[r, c]
                map_px[y1:y2, x1:x2] = val

        
        # --- 5) (optional) smooth slightly for aesthetics
        # Note: you can adjust or remove the blur; here light to avoid too sharp blocks
        sal = cv2.GaussianBlur(map_px, (0, 0), 10)
        sal = np.power(sal, 1.5)  # emphasize peaks
        sal_uint8 = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --- 6) Save .npy if requested
        if output_path:
            try:
                npy_path = output_path.replace('.png', '.npy')
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, sal_uint8)
            except Exception as e:
                print(f"Warning: failed to save saliency numpy array: {e}")

        # --- 7) Colormap + overlay
        heatmap_colored = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        return overlay_pil, sal_uint8

    
    def run_generation_collect_attentions(self, inputs, prompt_text=None):
        """Run generation to collect attentions and key metadata without computing heatmap."""
        # Find image token positions
        image_positions, num_image_tokens = self.find_image_token_positions(inputs, prompt_text)
        self.image_positions = image_positions
        self.image_tokens = num_image_tokens
        
        # Grid dimensions should already be set in prepare_inputs, but verify they match
       
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=200,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
                use_cache=True
            )

        input_length = inputs["input_ids"].size(1)
        self.len_input_ids = input_length
        generated_ids = outputs.sequences[0, input_length:]
        confidences = []
        for t, scores in enumerate(outputs.scores):
            probs = torch.softmax(scores[0], dim=-1)
            confidence = float(probs[generated_ids[t]].item())
            confidences.append(confidence)
        self.confidences = confidences
        self.attention = outputs.attentions
        return len(generated_ids)
    
    def attention_rollout_from_generation(self, attn_outputs,eps=1e-9):
        """
        Performs attention rollout starting from HuggingFace outputs.attentions
        obtained during autoregressive generation. For each generated token, for each layer average the attention between heads
        I do a rollout from the last to the first layer. When this model takes attention it returns a tensor [1,heads,n_input+1(prompt token generated), n_input+1(prompt token generated)]
        only for the first token. From the second it generates a tensor [1,heads,1,1,previous_token+1(generated token)]

        Args:
            attn_outputs: tuple/list
                - attn_outputs[0]: attention on inputs (before generation)
                    list of length num_layers
                    each element: tensor [1, num_heads, L, L]
                - attn_outputs[i] (i>=1): attention at step i of generation
                    list of length num_layers
                    each element: tensor [1, num_heads, 1, L+i]

            eps: float
                for numerical stability in normalization

        Returns:
            R: torch.Tensor [T, T]
            final rollout matrix (averaged heads and composed layers)
        """
        num_steps = len(attn_outputs) - 1     # number of generated tokens
        num_layers = len(attn_outputs[0])     # number of layers
        H, L, _ = attn_outputs[0][0].shape[1:]   # heads, seq_len_input, seq_len_input

        T = L + num_steps                     # final length (input + output)
        device = attn_outputs[0][0].device
        dtype = attn_outputs[0][0].dtype

        # Initialize rollout
        R = None
        A_rollout_final = None
        for step in range(num_steps+1):
            A_rollout = None
            for l in range(num_layers - 1, -1, -1):
                row = attn_outputs[step][l][0]  # [H, 1, L+step]
                seq_len = row.shape[2]
                row_mean = row.mean(0)
                if step != 0:
                    row_mean_zero = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
                    row_mean_zero[-1:] = row_mean 
                    row_mean = row_mean_zero
                I = torch.eye(seq_len, device=device, dtype=dtype)
                A = (row_mean + I)/2
                A = A / A.sum(dim=-1, keepdim=True)
                A_rollout = A if A_rollout is None else A_rollout @ A
            if step == 0:
                A_rollout_final = A_rollout
                continue
            else:
                A_rollout_final_padded = F.pad(A_rollout_final, (0, 1, 0, 1))
                A_rollout_final = A_rollout + A_rollout_final_padded
        return A_rollout_final

    def rollout_to_image_maps(self, R):
        """
        R: [T, T] from rollout 
        Always analyzes the attention of tokens present in R
        """
        num_tokens_in_R = R.shape[0]
        
        # Check if there are generated tokens in this sequence
        if num_tokens_in_R > self.len_input_ids:
            # There are generated tokens
            tokens_generated_so_far = num_tokens_in_R - self.len_input_ids
            # Get the attention of generated tokens on image patches
            per_token = R[self.len_input_ids:, self.image_positions]  # [tokens_gen, 576]
            num_gen = per_token.shape[0]
            maps_per_token = per_token.reshape(num_gen, 17, 13)
            # Aggregations
            map_mean = maps_per_token.mean(0)
            map_max = maps_per_token.max(0).values
            return maps_per_token, map_mean, map_max
            
        else:
            # Only initial input, no generated tokens
            # Use the attention of the last input token on image patches
            print("No generated tokens, using last input token")
            
            last_input_token_attention = R[self.len_input_ids-1, self.image_positions]  # [576]
            last_input_token_attention = last_input_token_attention / (last_input_token_attention.sum() + 1e-9)
            
            # Reshape
            attention_map = last_input_token_attention.reshape(17, 13)
        
            
        return attention_map.unsqueeze(0), attention_map, attention_map
            
        

    def analyze_image(self, image_path, prompt_text, output_path=None):
        """Complete analysis pipeline"""
        print("=== Starting Attention Analysis ===")
        
        # Prepare inputs
        inputs, raw_image = self.prepare_inputs(image_path, prompt_text)
        
        # Run generation to collect attentions (do not compute heatmap here)
        num_generated = self.run_generation_collect_attentions(inputs, prompt_text)
        print(f"Generated {num_generated} tokens with average confidence: {np.mean(self.confidences):.3f}")
        R = self.attention_rollout_from_generation(self.attention)
        maps_per_token, map_mean, map_max = self.rollout_to_image_maps(R)
        
        # Create overlay using the new Fuyu-specific method
        overlay_pil_rollout_mean, saliency_array_rollout_mean = self.create_attention_overlay_fuyu(
            map_mean, 
            raw_image, 
            patch_size=30,
            alpha=0.5,
            output_path=output_path.replace('.png', '_rollout_mean.png') if output_path else None
        )
        overlay_pil_rollout_max, saliency_array_rollout_max = self.create_attention_overlay_fuyu(
            map_max, 
            raw_image, 
            patch_size=30,
            alpha=0.5,
            output_path=output_path.replace('.png', '_rollout_max.png') if output_path else None
        )
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_path_rollout_mean = output_path.replace('.png', '_rollout_mean.png')
            overlay_pil_rollout_mean.save(output_path_rollout_mean)
            print(f"Overlay saved to: {output_path_rollout_mean}")
            output_path_rollout_max = output_path.replace('.png', '_rollout_max.png')
            overlay_pil_rollout_max.save(output_path_rollout_max)
            print(f"Overlay rollout saved to: {output_path_rollout_max}")
        
        return {
            'confidences': self.confidences,
            'num_generated': num_generated,
            'overlay': overlay_pil_rollout_mean,
            'saliency': saliency_array_rollout_mean,
            'overlay_rollout': overlay_pil_rollout_max,
            'saliency_rollout': saliency_array_rollout_max
        }


def main():
    """Main function with clean execution flow"""
    # Configuration
    import pandas as pd
    questions = pd.read_excel("/work/mmaz/mllm_study/data/prompt_files/user_1_session_1.xlsx")
    analyzer = AttentionAnalyzer()
    
    for i in range(1, 31):
        image_path = f"/work/mmaz/mllm_study/data/images/img_prompt_{i}.jpg"
        prompt_text = questions.loc[i-1, 'prompt_text']
        print(prompt_text)
        
        # Extract prompt number from image path
        import re
        prompt_match = re.search(r'img_prompt_(\d+)', image_path)
        if prompt_match:
            prompt_number = prompt_match.group(1)
            output_path = f"/work/mmaz/mllm_study/data/fuyu_attention/img_prompt_{prompt_number}.png"
        else:
            raise ValueError("Prompt number not found")
        
        try:
            # Run analysis
            results = analyzer.analyze_image(image_path, prompt_text, output_path)
            
            # Display results
            print("Analysis completed successfully!")
            del results
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


if __name__ == "__main__":
    main()
