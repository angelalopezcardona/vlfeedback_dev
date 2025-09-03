import torch
import numpy as np
from PIL import Image
import cv2
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from PIL import Image
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from utils import (
    load_image, 
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)

model_id = "llava-hf/llava-1.5-13b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    attn_implementation="eager"
).to(1)
'''
print('ciao')
print("Sotto-moduli top-level:", [n for n, _ in model.named_children()])
print("Ha vision_model?", hasattr(model, "vision_tower"))
print("Ha multi_modal_projector?", hasattr(model, "multi_modal_projector"))
print("Ha language_model?", hasattr(model, "language_model"))
num_patches = model.config.image_seq_length
print("Numero di patch visivi:", num_patches)
print("Config keys:", list(model.config.to_dict().keys()))
'''
processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "image"},
          {"type": "text", "text": "ciao?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(repr(prompt))               # vedrai qualcosa tipo "USER: <image>\nProvide ...\nASSISTANT:"

image_file = "/work/mmaz/mllm_study/data/images/img_prompt_4.jpg"
raw_image = Image.open(image_file).convert('RGB')
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(1, torch.float16)

outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=200,
    return_dict_in_generate=True,
    output_attentions=True,
    output_scores=True,
)
print(len(inputs["input_ids"][0]))

'''
for tid in inputs["input_ids"][0][597:582]:  # primi 20 token
    print(tid.item(), "->", processor.tokenizer.decode([tid.item()]))
'''
vision_tower = model.model.vision_tower
vision_tower.config.output_attentions = True
with torch.no_grad():
    vision_outputs = vision_tower(
        pixel_values=inputs["pixel_values"],
        output_attentions=True,
        return_dict=True
    )

vit_attn= vision_outputs.attentions  # lista [24] di (batch, heads, tokens, tokens)

vision_token_start = len(processor.tokenizer(repr(prompt).split("<image>")[0], return_tensors='pt')["input_ids"][0])
vision_token_end = vision_token_start + 576
#print(processor.decode(outputs["sequences"][0], skip_special_tokens=True))
aggregated_prompt_attention = []
for i, layer in enumerate(outputs["attentions"][0]):
    layer_attns = layer.squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    cur = attns_per_head[:-1].cpu().clone()
    # following the practice in `aggregate_llm_attention`
    # we are zeroing out the attention to the first <bos> token
    # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
    # we don't do this because <bos> is the only token that it can attend to
    cur[1:, 0] = 0.
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    aggregated_prompt_attention.append(cur)
aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
print(aggregated_prompt_attention.shape)
llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention) 
    + list(map(aggregate_llm_attention, outputs["attentions"]))
)
print(llm_attn_matrix.shape)
vis_attn_matrix = aggregate_vit_attention(
    vit_attn,
    select_layer=-2,
    all_prev_layers=True)
output_token_start = aggregated_prompt_attention.shape[0]+1
output_token_end = llm_attn_matrix.shape[0]

def create_unified_attention_overlay(llm_attn_matrix, vis_attn_matrix, 
                                   output_token_start, output_token_end,
                                   vision_token_start, vision_token_end, 
                                   grid_size, image, method='mean'):
    """
    Crea un overlay unico dell'attenzione per tutta la risposta generata
    
    Args:
        method: 'mean' | 'weighted_mean' | 'max' | 'sum'
    """
    
    # Estrai tutti i token generati (con controllo bounds)
    matrix_size = llm_attn_matrix.shape[0]
    output_token_end_safe = min(output_token_end, matrix_size)
    output_token_inds = list(range(output_token_start, output_token_end_safe))
    
    if output_token_end > matrix_size:
        print(f"Warning: output_token_end ({output_token_end}) > matrix_size ({matrix_size}). Usando {output_token_end_safe}")
    
    all_attentions = []
    
    for token_ind in output_token_inds:
        # Attenzione di questo token verso i patch visivi
        attn_weights_over_vis_tokens = llm_attn_matrix[token_ind][vision_token_start:vision_token_end]
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()
        
        # Propaga attraverso l'attenzione ViT
        attn_over_image = []
        for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
            vis_attn = vis_attn.reshape(grid_size, grid_size)
            attn_over_image.append(vis_attn * weight)
        
        attn_over_image = torch.stack(attn_over_image).sum(dim=0)
        all_attentions.append(attn_over_image)
    
    # Aggregazione secondo il metodo scelto
    all_attentions = torch.stack(all_attentions)
    
    if method == 'mean':
        unified_attention = all_attentions.mean(dim=0)
    elif method == 'weighted_mean':
        # Pesa di più i token "importanti" (con più attenzione totale)
        weights = all_attentions.sum(dim=(1,2))  # Attenzione totale per token
        weights = weights / weights.sum()
        unified_attention = (all_attentions * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
    elif method == 'max':
        unified_attention = all_attentions.max(dim=0)[0]
    elif method == 'sum':
        unified_attention = all_attentions.sum(dim=0)
    
    # Normalizza
    unified_attention = unified_attention / unified_attention.max()
    
    # Interpola alla dimensione dell'immagine (correggi ordine height, width)
    img_array = np.array(image)
    target_size = img_array.shape[:2]  # (height, width) invece di image.size che è (width, height)
    
    unified_attention = F.interpolate(
        unified_attention.unsqueeze(0).unsqueeze(0), 
        size=target_size, 
        mode='nearest'
    ).squeeze()
    
    return unified_attention
def save_unified_attention_overlay(llm_attn_matrix, vis_attn_matrix,
                                 output_token_start, output_token_end,
                                 vision_token_start, vision_token_end, 
                                 grid_size, image, 
                                 method='weighted_mean',
                                 output_dir='./attention_outputs/',
                                 save_all_methods=False):
    """
    Crea e salva overlay di attenzione unificata
    
    Args:
        method: 'mean' | 'weighted_mean' | 'max' | 'sum' | 'all'
        output_dir: Directory dove salvare i file
        save_all_methods: Se True, salva tutti i metodi (ignora 'method')
    """
    import os
    
    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Determina quali metodi salvare
    if save_all_methods or method == 'all':
        methods_to_save = ['mean', 'weighted_mean', 'max', 'sum']
    else:
        methods_to_save = [method]
    
    results = {}
    
    for current_method in methods_to_save:
        # Calcola attenzione unificata
        unified_attention = create_unified_attention_overlay(
            llm_attn_matrix, vis_attn_matrix,
            output_token_start, output_token_end,
            vision_token_start, vision_token_end,
            grid_size, image, 
            method=current_method
        )

        # Visualizzazione
        plt.figure(figsize=(12, 8))

        # Prepara immagine numpy
        np_img = np.array(image)[:, :, ::-1]  # RGB to BGR
        print(f"Debug - np_img shape: {np_img.shape}")
        print(f"Debug - unified_attention shape: {unified_attention.shape}")
        
        # Assicurati che le dimensioni coincidano
        if np_img.shape[:2] != unified_attention.shape:
            print(f"Warning: Reshaping attention from {unified_attention.shape} to {np_img.shape[:2]}")
            unified_attention = F.interpolate(
                unified_attention.unsqueeze(0).unsqueeze(0),
                size=np_img.shape[:2],  # (height, width)
                mode='nearest'
            ).squeeze()
        
        # Subplot 1: Solo heatmap
        plt.subplot(1, 2, 1)
        _, heatmap = show_mask_on_image(np_img, unified_attention.numpy().astype(np.float32))
        plt.imshow(heatmap)
        plt.title(f"Heatmap Attenzione - {current_method.title()}", fontsize=14)
        plt.axis('off')

        # Subplot 2: Overlay con immagine
        plt.subplot(1, 2, 2)
        img_with_attn, _ = show_mask_on_image(np_img, unified_attention.numpy().astype(np.float32))
        plt.imshow(img_with_attn)
        plt.title(f"Immagine + Overlay - {current_method.title()}", fontsize=14)
        plt.axis('off')

        plt.tight_layout()
        
        # Salva con nome che include il metodo
        filename = f"attention_overlay_{current_method}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Chiude la figura per liberare memoria
        
        print(f"Salvato: {filepath}")
        results[current_method] = {'attention': unified_attention, 'file': filepath}
    
    # Ritorna risultato singolo se solo un metodo, altrimenti dizionario completo
    if len(results) == 1:
        method_name = list(results.keys())[0]
        return results[method_name]['attention'], results[method_name]['file']
    else:
        print(f"\nSalvati {len(results)} file in: {output_dir}")
        return results

all_results = save_unified_attention_overlay(
    llm_attn_matrix, vis_attn_matrix,
    output_token_start, output_token_end,
     vision_token_start, vision_token_end,
     24, raw_image,
    method='mean'  # oppure save_all_methods=True
    )
'''
class LLaVAAttentionAnalyzer:
    def __init__(self, model_name="llava-hf/llava-1.5-13b-hf"):
        """
        Inizializza l'analizzatore di attenzione per LLaVA-1.5
        """
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f"Caricamento modello su {self.device}...")
        
        # Carica il processore e il modello per LLaVA-1.5
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager"  # Forza l'implementazione eager per ottenere i pesi di attenzione
        )
        
        # Hook per catturare i pesi di attenzione
        self.attention_weights = {}
        self.vision_attention_weights = []
        self.cross_attention_weights = []
        self.register_hooks()
    
    def register_hooks(self):
        """Registra gli hook per catturare i pesi di attenzione"""
        def vision_attention_hook(name):
            def hook(module, input, output):
                # Per i layer di self-attention della vision tower
                if hasattr(output, 'attentions') and output.attentions is not None:
                    self.vision_attention_weights.append(output.attentions.detach())
                elif isinstance(output, tuple) and len(output) > 1:
                    # Alcuni modelli restituiscono (hidden_states, attention_weights)
                    if isinstance(output[1], torch.Tensor) and len(output[1].shape) == 4:
                        self.vision_attention_weights.append(output[1].detach())
            return hook
        
        def cross_attention_hook(name):
            def hook(module, input, output):
                # Per i layer di cross-attention tra vision e text
                if hasattr(output, 'cross_attentions') and output.cross_attentions is not None:
                    self.cross_attention_weights.append(output.cross_attentions.detach())
                elif hasattr(output, 'attentions') and output.attentions is not None:
                    self.cross_attention_weights.append(output.attentions.detach())
            return hook
        
        # Registra hook sui layer del vision encoder
        for name, module in self.model.named_modules():
            if 'vision_tower' in name and 'attention' in name:
                module.register_forward_hook(vision_attention_hook(name))
            elif 'multi_modal_projector' in name:
                module.register_forward_hook(cross_attention_hook(name))
    
    def process_image_and_prompt(self, image_path, prompt):
        """Processa l'immagine e il prompt per LLaVA-1.5"""
        # Carica e preprocessa l'immagine
        image = Image.open(image_path).convert('RGB')
        
        # Formato del prompt per LLaVA-1.5
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Applica il template di chat
        formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Prepara gli input
        inputs = self.processor(
            images=image, 
            text=formatted_prompt, 
            return_tensors="pt"
        )
        
        # Sposta su device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        return image, inputs
    
    def generate_with_attention(self, inputs):
        """Genera la risposta catturando i pesi di attenzione"""
        self.attention_weights.clear()
        self.vision_attention_weights.clear()
        self.cross_attention_weights.clear()
        
        with torch.no_grad():
            # Genera la risposta
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        return outputs
    
    def extract_vision_features_attention(self):
        """Estrae l'attenzione dalle feature vision usando gradient-based saliency"""
        # Metodo alternativo: usa i gradienti per la saliency
        if not hasattr(self, 'last_vision_features'):
            return None
            
        # Calcola i gradienti rispetto alle feature vision
        vision_features = self.last_vision_features
        if vision_features.requires_grad:
            vision_features.retain_grad()
            
        return vision_features
    
    def create_gradient_saliency(self, inputs, target_token_id=None):
        """Crea una mappa di saliency usando i gradienti"""
        # Abilita i gradienti per l'immagine
        pixel_values = inputs['pixel_values'].clone().detach().requires_grad_(True)
        modified_inputs = {k: v for k, v in inputs.items()}
        modified_inputs['pixel_values'] = pixel_values
        
        # Forward pass
        outputs = self.model(**modified_inputs, output_attentions=True)
        
        # Usa l'ultimo logit come target se non specificato
        if target_token_id is None:
            target_logit = outputs.logits[0, -1, :].max()
        else:
            target_logit = outputs.logits[0, -1, target_token_id]
        
        # Backward pass
        target_logit.backward()
        
        # Ottieni i gradienti
        gradients = pixel_values.grad.abs()
        
        # Aggrega sui canali
        saliency = gradients.squeeze(0).mean(dim=0).cpu().numpy()
        
        return saliency
    
    def create_attention_rollout(self, attention_matrices):
        """Implementa Attention Rollout per aggregare le attenzioni multi-layer"""
        if not attention_matrices:
            return None
            
        # Converti tutti in numpy e prendi la media sui heads
        processed_attentions = []
        for att in attention_matrices:
            if isinstance(att, torch.Tensor):
                att = att.cpu().numpy()
            if len(att.shape) == 4:  # [batch, heads, seq, seq]
                att = att.mean(axis=1)  # Media sui heads
            if len(att.shape) == 3:  # [batch, seq, seq]
                att = att[0]  # Prendi il primo batch
            processed_attentions.append(att)
        
        if not processed_attentions:
            return None
            
        # Implementa attention rollout
        result = processed_attentions[0]
        for att in processed_attentions[1:]:
            result = np.matmul(result, att)
        
        return result
    
    def create_saliency_from_gradient(self, gradient_saliency, original_image):
        """Converte la saliency basata sui gradienti in una mappa visualizzabile"""
        # Normalizza
        saliency = (gradient_saliency - gradient_saliency.min()) / \
                  (gradient_saliency.max() - gradient_saliency.min() + 1e-8)
        
        # Ridimensiona se necessario
        img_array = np.array(original_image)
        h, w = img_array.shape[:2]
        
        if saliency.shape != (h, w):
            saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return saliency
    
    def visualize_attention(self, image, saliency_map, save_path=None, method_name="Gradient Saliency"):
        """Visualizza l'immagine originale e la mappa di saliency"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Immagine originale
        axes[0].imshow(image)
        axes[0].set_title('Immagine Originale')
        axes[0].axis('off')
        
        # Mappa di saliency
        im1 = axes[1].imshow(saliency_map, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f'Mappa di Saliency ({method_name})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(image, alpha=0.7)
        axes[2].imshow(saliency_map, cmap='hot', alpha=0.4, interpolation='bilinear')
        axes[2].set_title('Overlay Attenzione')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizzazione salvata in: {save_path}")
        
        plt.show()
    
    def analyze_attention(self, image_path, prompt, save_visualization=True):
        """Funzione principale per l'analisi completa"""
        print("Caricamento e preprocessing dell'immagine...")
        image, inputs = self.process_image_and_prompt(image_path, prompt)
        
        print("Generazione della risposta...")
        outputs = self.generate_with_attention(inputs)
        
        # Decodifica la risposta
        generated_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
        # Rimuovi il prompt dalla risposta
        if prompt in generated_text:
            response = generated_text.split(prompt)[-1].strip()
        else:
            response = generated_text.strip()
            
        print(f"\nRisposta del modello:\n{response}")
        
        print("\nGenerazione mappa di saliency con metodo gradient-based...")
        try:
            gradient_saliency = self.create_gradient_saliency(inputs)
            saliency_map = self.create_saliency_from_gradient(gradient_saliency, image)
            
            if save_visualization:
                save_path = image_path.replace('.jpg', '_attention_analysis.png')
                self.visualize_attention(image, saliency_map, save_path)
            else:
                self.visualize_attention(image, saliency_map)
            
            return {
                'response': response,
                'saliency_map': saliency_map,
                'gradient_saliency': gradient_saliency
            }
            
        except Exception as e:
            print(f"Errore nella generazione della saliency: {e}")
            print("Provo con metodo alternativo...")
            
            # Metodo alternativo: attention rollout se disponibile
            if self.vision_attention_weights or self.cross_attention_weights:
                all_attentions = self.vision_attention_weights + self.cross_attention_weights
                attention_result = self.create_attention_rollout(all_attentions)
                
                if attention_result is not None:
                    # Crea una saliency semplificata
                    saliency_simple = np.mean(attention_result, axis=0)
                    saliency_simple = (saliency_simple - saliency_simple.min()) / \
                                    (saliency_simple.max() - saliency_simple.min() + 1e-8)
                    
                    # Ridimensiona all'immagine
                    img_array = np.array(image)
                    h, w = img_array.shape[:2]
                    saliency_resized = cv2.resize(saliency_simple, (w, h))
                    
                    if save_visualization:
                        save_path = image_path.replace('.jpg', '_attention_rollout.png')
                        self.visualize_attention(image, saliency_resized, save_path, "Attention Rollout")
                    else:
                        self.visualize_attention(image, saliency_resized, method_name="Attention Rollout")
                    
                    return {
                        'response': response,
                        'saliency_map': saliency_resized,
                        'attention_rollout': attention_result
                    }
        
        return {'response': response}

# Utilizzo
if __name__ == "__main__":
    # Inizializza l'analizzatore
    analyzer = LLaVAAttentionAnalyzer("llava-hf/llava-1.5-13b-hf")
    
    # Parametri
    image_path = "/work/mmaz/mllm_study/data/images/img_prompt_4.jpg"
    prompt = "describe the image in detail"
    
    # Esegui l'analisi
    results = analyzer.analyze_attention(image_path, prompt)
    
    print("\nAnalisi completata!")
    if 'saliency_map' in results:
        print("Mappa di saliency generata con successo!")
    else:
        print("Non è stato possibile generare la mappa di saliency.")
'''