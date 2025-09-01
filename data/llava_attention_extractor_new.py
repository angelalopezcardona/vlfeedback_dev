import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import cv2
import os
from scipy.ndimage import zoom


class AttentionAnalyzer:
    """Class to handle attention analysis for LLaVA model"""
    
    def __init__(self, model_id="llava-hf/llava-1.5-13b-hf", device_id=1):
        """Initialize the attention analyzer with model and processor"""
        self.model_id = model_id
        self.device_id = device_id
        self.model = None
        self.processor = None
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
        print(f"Found {num_image_tokens} image tokens")
        
        # Calculate grid size (assumes square)
        grid_size = int(np.sqrt(num_image_tokens))
        if grid_size * grid_size != num_image_tokens:
            print(f"Warning: {num_image_tokens} is not a perfect square")
            grid_size = int(np.sqrt(num_image_tokens))
        
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
        generated_ids = outputs.sequences[0, input_length:]
        num_generated = len(generated_ids)
        
        # Calculate confidence for each generated token
        confidences = []
        for t, scores in enumerate(outputs.scores):
            probs = torch.softmax(scores[0], dim=-1)
            confidence = float(probs[generated_ids[t]].item())
            confidences.append(confidence)
        
        # Process attention efficiently
        device = inputs["input_ids"].device
        accumulated_attention = torch.zeros(num_image_tokens, dtype=torch.float32, device=device)
        total_weight = 0.0
        
        lambda_d = 0.3  # Depth prior
        
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
    
    def create_attention_overlay(self, heatmap_24x24, pil_image, alpha=0.5, sigma=30):
        """Create attention overlay directly with PIL Image"""
        
        # Convert PIL → OpenCV
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        height, width = image_cv.shape[:2]
        
        # Upscale
        scale_y, scale_x = height / 24, width / 24
        upscaled = zoom(heatmap_24x24, (scale_y, scale_x), order=1)
        
        # Processing
        saliency = (upscaled - upscaled.min()) / (upscaled.max() - upscaled.min() + 1e-8)
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), sigma)
        saliency = np.power(saliency, 1.5)
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Overlay
        heatmap_colored = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        overlay_cv = cv2.addWeighted(image_cv, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Convert OpenCV → PIL
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay_cv, cv2.COLOR_BGR2RGB))
        
        return overlay_pil, saliency
    
    def analyze_image(self, image_path, prompt_text, output_path=None):
        """Complete analysis pipeline"""
        print("=== Starting Attention Analysis ===")
        
        # Prepare inputs
        inputs, raw_image = self.prepare_inputs(image_path, prompt_text)
        
        # Compute attention heatmap
        heatmap, confidences, num_generated = self.compute_attention_heatmap(inputs)
        print(f"Generated {num_generated} tokens with average confidence: {np.mean(confidences):.3f}")
        
        # Create overlay
        overlay_pil, saliency_array = self.create_attention_overlay(
            heatmap, 
            raw_image, 
            alpha=0.5
        )
        
        # Save results
        if output_path:
            overlay_pil.save(output_path)
            print(f"🎉 Overlay saved to: {output_path}")
        
        return {
            'heatmap': heatmap,
            'confidences': confidences,
            'num_generated': num_generated,
            'overlay': overlay_pil,
            'saliency': saliency_array
        }


def main():
    """Main function with clean execution flow"""
    # Configuration
    image_path = "/work/mmaz/mllm_study/data/images/img_prompt_1.jpg"
    prompt_text = "what is the name of the school mentioned in the image?"
    
    # Extract prompt number from image path
    import re
    prompt_match = re.search(r'img_prompt_(\d+)', image_path)
    if prompt_match:
        prompt_number = prompt_match.group(1)
        output_path = f"/work/mmaz/mllm_study/data/attention_outputs/llava_overlay_img_prompt_{prompt_number}.png"
    else:
        output_path = "/work/mmaz/mllm_study/data/attention_outputs/llava_overlay_img_prompt_1.png"
    
    try:
        # Initialize analyzer
        analyzer = AttentionAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_image(image_path, prompt_text, output_path)
        
        # Display results
        print("Analysis completed successfully!")
        print(f"Generated {results['num_generated']} tokens")
        print(f"Average confidence: {np.mean(results['confidences']):.3f}")
        
        # Show the overlay
        results['overlay'].show()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
