import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import cv2
import os
from scipy.ndimage import zoom


class QwenVLAttentionAnalyzer:
    """Class to handle attention analysis for Qwen-VL-Chat model"""
    
    def __init__(self, model_id="Qwen/Qwen-VL-Chat", device_id="cuda:1"):
        """Initialize the attention analyzer with model and tokenizer"""
        self.model_id = model_id
        self.device_id = device_id
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen-VL-Chat model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map=self.device_id, 
            trust_remote_code=True,
            use_cache=False,
        ).eval()
        
        # Configurazione per la generazione
        self.model.generation_config = GenerationConfig.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            use_cache=False
        )
    
    def prepare_inputs(self, image_path, prompt_text):
        """Prepare inputs for the model"""
        # Prepara input con formato conversazionale
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt_text},
        ])
        
        # Aggiungi i token di sistema per formato conversazionale
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        assistant_prompt = "<|im_end|>\n<|im_start|>assistant\n"
        
        full_prompt = system_prompt + query + assistant_prompt
        
        # Tokenizza
        inputs = self.tokenizer(full_prompt, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        
        # Carica l'immagine per l'overlay
        raw_image = Image.open(image_path).convert('RGB')
        
        return inputs, raw_image
    
    def find_image_token_positions(self, inputs):
        """Find the positions of <imgpad> tokens in the sequence"""
        tok = self.tokenizer
        print(tok.convert_ids_to_tokens([151859]))
        # Find image token ID for <imgpad>
        image_token_id = None
        image_token_id = tok.convert_tokens_to_ids("<imgpad>")

        if image_token_id == tok.unk_token_id:  # non trovato
            raise ValueError("Token <imgpad> not found")
        input_ids = inputs["input_ids"][0]
        
        # Count tokens with ID 151859
        tokens_151859 = (input_ids == 151859).sum().item()
        print(f"Numero di token con ID 151859 negli input: {tokens_151859}")
        
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
        
        # Custom stopping criteria
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class CustomStoppingCriteria(StoppingCriteria):
            def __init__(self, tokenizer):
                self.im_end_id = tokenizer.special_tokens["<|im_end|>"]
                self.im_start_id = tokenizer.special_tokens["<|im_start|>"]
                
            def __call__(self, input_ids, scores, **kwargs):
                # Fermati se trova <|im_end|> o <|im_start|> (inizio nuova conversazione)
                last_token = input_ids[0][-1].item()
                return last_token == self.im_end_id or last_token == self.im_start_id
        
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(self.tokenizer)])
        
        # Generate with attention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
                use_cache=False
            )
        
        # Extract generated tokens and confidence
        input_length = inputs["input_ids"].size(1)
        generated_ids = outputs.sequences[0, input_length:]
        num_generated = len(generated_ids)
        
        # Decode and print the generated response
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Risposta generata: {generated_text}")
        
        # Get token details (ID and text conversion) for each generated token
        token_details = []
        for i, token_id in enumerate(generated_ids):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            token_details.append({
                'id': token_id.item(),
                'text': token_text,
                'position': i
            })
        
        # Print token details
        print("\nDettagli dei token generati:")
        for token in token_details:
            print(f"Posizione {token['position']}: ID={token['id']}, Testo='{token['text']}'")
        
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
        
        attn_np = final_attention.cpu().numpy()

        # --- 🔑 Adattamento a griglia 14x15 (210 slot, con padding) ---
        h, w = 14, 15
        padded = np.zeros(h * w, dtype=np.float32)
        padded[:num_image_tokens] = attn_np  # inserisce i 206 valori nei primi slot
        heatmap = padded.reshape(h, w)
        test = np.arange(num_image_tokens)
        padded_test = np.zeros(h * w)
        padded_test[:num_image_tokens] = test

        return heatmap, confidences, num_generated, token_details
    
    def create_attention_overlay(self, heatmap, pil_image, alpha=0.5, sigma=30):
        """Create attention overlay directly with PIL Image"""
        
        # Convert PIL → OpenCV
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        height, width = image_cv.shape[:2]
        
        # Get grid dimensions from heatmap shape (h, w)
        grid_h, grid_w = heatmap.shape
        
        # Upscale to match image dimensions
        scale_y, scale_x = height / grid_h, width / grid_w
        upscaled = zoom(heatmap, (scale_y, scale_x), order=1)
        
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
        print("=== Starting Qwen-VL Attention Analysis ===")
        
        # Prepare inputs
        inputs, raw_image = self.prepare_inputs(image_path, prompt_text)
        
        # Compute attention heatmap
        heatmap, confidences, num_generated, token_details = self.compute_attention_heatmap(inputs)
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
            'token_details': token_details,
            'overlay': overlay_pil,
            'saliency': saliency_array
        }


def main():
    """Main function with clean execution flow"""
    # Configuration
    image_path = "/work/mmaz/mllm_study/data/images/img_prompt_2.jpg"
    prompt_text = "Can you spot the purple umbrella near the green leaves? It appears to be left behind by someone.?"
    
    # Extract prompt number from image path
    import re
    prompt_match = re.search(r'img_prompt_(\d+)', image_path)
    if prompt_match:
        prompt_number = prompt_match.group(1)
        output_path = f"/work/mmaz/mllm_study/data/attention_outputs/qwen_overlay_img_prompt_{prompt_number}.png"
    else:
        output_path = "/work/mmaz/mllm_study/data/attention_outputs/qwen_overlay_img_prompt_1.png"
    
    try:
        # Initialize analyzer
        analyzer = QwenVLAttentionAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_image(image_path, prompt_text, output_path)
        
        # Display results
        print("Analysis completed successfully!")
        print(f"Generated {results['num_generated']} tokens")
        print(f"Average confidence: {np.mean(results['confidences']):.3f}")
        
        # Display token details summary
        print(f"\nRiepilogo token generati:")
        for token in results['token_details']:
            print(f"  Token {token['position']}: ID={token['id']}, Testo='{token['text']}'")
        
        # Show the overlay
        results['overlay'].show()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()