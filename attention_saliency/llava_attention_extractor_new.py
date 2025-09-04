import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import cv2
import os
from scipy.ndimage import zoom
import torch.nn.functional as F
import pandas as pd


class AttentionAnalyzer:
    """Class to handle attention analysis for LLaVA model"""
    
    def __init__(self, model_id="llava-hf/llava-1.5-13b-hf", device_id=1):
        """Initialize the attention analyzer with model and processor"""
        self.model_id = model_id
        self.device_id = device_id
        self.model = None
        self.processor = None
        self.attention = None
        self.len_input_ids = None
        self.image_positions = None
        self.image_tokens = None
        self.grid_size = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLaVA model and processor"""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            attn_implementation="eager"
        ).to(self.device_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def prepare_inputs(self, image_path, prompt_text):
        """Prepare inputs for the model"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt').to(self.device_id, torch.float16)
        
        return inputs, raw_image
    
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
    
    def compute_attention_heatmap(self, inputs):
        """Compute attention heatmap efficiently with a single forward pass"""
        
        # Find image token positions
        image_positions, num_image_tokens = self.find_image_token_positions(inputs)
        self.image_positions = image_positions
        self.image_tokens = num_image_tokens
        print(f"Found {num_image_tokens} image tokens")
        
        # Calculate grid size (assumes square)
        grid_size = int(np.sqrt(num_image_tokens))
        self.grid_size = grid_size
        if grid_size * grid_size != num_image_tokens:
            print(f"Warning: {num_image_tokens} is not a perfect square")
            grid_size = int(np.sqrt(num_image_tokens))
            self.grid_size = grid_size
        
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
        
        # Move to CPU and reshape
        heatmap = final_attention.cpu().numpy().reshape(grid_size, grid_size)
        
        return heatmap, confidences, num_generated
    
    def create_attention_overlay(self, heatmap_24x24, original_pil_image, model_input_size=336, alpha=0.6, output_path=None):
        """
        Create attention overlay aligned with the ORIGINAL image.
        
        Args:
            heatmap_24x24: The 24x24 attention map from the model.
            original_pil_image: The original PIL image (any size).
            model_input_size: The size the model expects (e.g., 336 for LLaVA).
            alpha: Transparency for the overlay.
        """
        # Convert original PIL → OpenCV
        original_image_cv = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
        original_h, original_w = original_image_cv.shape[:2]
        
        # 1. Convert heatmap to numpy
        if isinstance(heatmap_24x24, torch.Tensor):
            heatmap_np = heatmap_24x24.detach().float().cpu().numpy()
        else:
            heatmap_np = np.asarray(heatmap_24x24, dtype=np.float32)
        
        # 2. Upscale the 24x24 heatmap to the MODEL'S INPUT SIZE (336x336) using NEAREST
        # This gives us the heatmap on the cropped, square image.
        model_size_heatmap = cv2.resize(
            heatmap_np,
            (model_input_size, model_input_size), # 336x336
            interpolation=cv2.INTER_NEAREST # USING NEAREST IS FUNDAMENTAL
        )
        
        # 3. Now we need to "reverse the transformation" applied to the original image.
        # This is the CRITICAL PART.
        
        # Step 3a: Resize the original image to have its short side = model_input_size (336)
        scale_factor = model_input_size / min(original_w, original_h)
        new_w = int(original_w * scale_factor)
        new_h = int(original_h * scale_factor)
        resized_image = cv2.resize(original_image_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Step 3b: Create a BLACK CANVAS the size of the RESIZED image (e.g., 544x336)
        # This canvas will hold our heatmap in the correct position.
        resized_heatmap_canvas = np.zeros((new_h, new_w), dtype=np.float32)
        
        # Step 3c: Calculate the coordinates for the CENTER CROP on the resized image
        start_x = max(0, (new_w - model_input_size) // 2)
        start_y = max(0, (new_h - model_input_size) // 2)

        end_x = min(start_x + model_input_size, new_w)
        end_y = min(start_y + model_input_size, new_h)

        crop_w = end_x - start_x
        crop_h = end_y - start_y

        # Step 3d: Place the model's heatmap (336x336) onto the black canvas at the crop position
        resized_heatmap_canvas[start_y:end_y, start_x:end_x] = model_size_heatmap[:crop_h, :crop_w]
        
        
        
        # Step 3e: Now, resize this LARGE heatmap canvas back to the ORIGINAL image size
        # This is the heatmap aligned with the original image!
        final_heatmap = cv2.resize(
            resized_heatmap_canvas,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST # Keep using NEAREST!
        )
        
        # 4. Normalize the final heatmap to [0, 1]
        saliency = (final_heatmap - final_heatmap.min()) / (final_heatmap.max() - final_heatmap.min() + 1e-8)
        
        # 5. Apply a Gaussian blur to smooth the blockyness (optional, but looks better)
        # The sigma should be proportional to the original image size.
        sigma = max(1, int(min(original_w, original_h) / 100)) # Adaptive sigma
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), sigmaX=60)
        saliency = np.power(saliency, 1.5)
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Persist normalized saliency if an output path is provided
        if output_path is not None:
            try:
                npy_path = output_path.replace('.png', '.npy')
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, saliency)
            except Exception as save_err:
                print(f"Warning: failed to save saliency numpy array to {output_path}: {save_err}")
        # 6. Convert the heatmap to a color map (e.g., Jet)
        heatmap_colored = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        
        # 7. Superimpose the heatmap onto the original OpenCV image
        overlay = original_image_cv.copy()
        overlay = cv2.addWeighted(overlay, 1 - alpha, heatmap_colored, alpha, 0)
        
        # 8. Convert back to PIL if needed
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        
        return overlay_pil, saliency
    
    def run_generation_collect_attentions(self, inputs):
        """Run generation to collect attentions and key metadata without computing heatmap."""
        # Find image token positions
        image_positions, num_image_tokens = self.find_image_token_positions(inputs)
        self.image_positions = image_positions
        self.image_tokens = num_image_tokens
        grid_size = int(np.sqrt(num_image_tokens))
        self.grid_size = grid_size
        if grid_size * grid_size != num_image_tokens:
            print(f"Warning: {num_image_tokens} is not a perfect square")
            grid_size = int(np.sqrt(num_image_tokens))
            self.grid_size = grid_size

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
            print(f"Generated tokens found: {tokens_generated_so_far}")
            
            # Get the attention of generated tokens on image patches
            per_token = R[self.len_input_ids:, self.image_positions]  # [tokens_gen, 576]
            num_gen = per_token.shape[0]
            maps_per_token = per_token.reshape(num_gen, self.grid_size, self.grid_size)
            
            # Aggregations
            map_mean = maps_per_token.mean(0)
            map_max = maps_per_token.max(0).values
            #print(map_mean)
            
            return maps_per_token, map_mean, map_max
            
        else:
            # Only initial input, no generated tokens
            # Use the attention of the last input token on image patches
            print("No generated tokens, using last input token")
            
            last_input_token_attention = R[self.len_input_ids-1, self.image_positions]  # [576]
            last_input_token_attention = last_input_token_attention / (last_input_token_attention.sum() + 1e-9)
            
            # Reshape
            attention_map = last_input_token_attention.reshape(self.grid_size, self.grid_size)
            
        return attention_map.unsqueeze(0), attention_map, attention_map

    def analyze_image(self, image_path, prompt_text, output_path=None):
        """Complete analysis pipeline"""
        print("=== Starting Attention Analysis ===")
        
        # Prepare inputs
        inputs, raw_image = self.prepare_inputs(image_path, prompt_text)
        
        # Run generation to collect attentions (do not compute heatmap here)
        num_generated = self.run_generation_collect_attentions(inputs)
        print(f"Generated {num_generated} tokens with average confidence: {np.mean(self.confidences):.3f}")
        R = self.attention_rollout_from_generation(self.attention)
        maps_per_token, map_mean, map_max = self.rollout_to_image_maps(R)
        


        
        # Create overlay
        overlay_pil_rollout_mean, saliency_array_rollout_mean = self.create_attention_overlay(
            map_mean, 
            raw_image, 
            alpha=0.5,
            output_path=output_path.replace('.png', '_rollout_mean.png') if output_path else None
        )
        overlay_pil_rollout_max, saliency_array_rollout_max = self.create_attention_overlay(
            map_max, 
            raw_image, 
            alpha=0.5,
            output_path=output_path.replace('.png', '_rollout_max.png') if output_path else None
        )
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_path_rollout_mean = output_path.replace('.png', '_rollout_mean.png')
            overlay_pil_rollout_mean.save(output_path_rollout_mean)
            print(f"🎉 Overlay saved to: {output_path_rollout_mean}")
            output_path_rollout_max = output_path.replace('.png', '_rollout_max.png')
            overlay_pil_rollout_max.save(output_path_rollout_max)
            print(f"🎉 Overlay rollout saved to: {output_path_rollout_max}")
        
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
    
    questions = pd.read_excel("/work/mmaz/mllm_study/data/prompt_files/user_1_session_1.xlsx")
    analyzer = AttentionAnalyzer()
    for i in range(1,31):
        image_path = f"/work/mmaz/mllm_study/data/images/img_prompt_{i}.jpg"
        prompt_text = questions.loc[i-1, 'prompt_text']
        print(prompt_text)
        # Extract prompt number from image path
        import re
        prompt_match = re.search(r'img_prompt_(\d+)', image_path)
        if prompt_match:
            prompt_number = prompt_match.group(1)
            output_path = f"/work/mmaz/mllm_study/data/llava_attention/img_prompt_{prompt_number}.png"
        else:
            raise ValueError("Prompt number not found")
        
        try:
            # Initialize analyzer
            
            
            # Run analysis
            results = analyzer.analyze_image(image_path, prompt_text, output_path)
            #analyzer.debug_attention(analyzer.attention)
            # Display results
            print("Analysis completed successfully!")
            print(f"Generated {results['num_generated']} tokens")
            print(f"Average confidence: {np.mean(results['confidences']):.3f}")
            
            # Show the overlay
            results['overlay'].show()
            results['overlay_rollout'].show()
            del  results
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


if __name__ == "__main__":
    main()
