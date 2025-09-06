import torch
from llava.mm_utils import process_images, tokenizer_image_token
from PIL import Image
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from llava.conversation import conv_templates
from peft import PeftModel
######### IMPORTANT #########
#before you need to install github library
#pip install git+https://github.com/haotian-liu/LLaVA.git


# Download the model from HuggingFace Hub
snapshot_download(
    repo_id="zhiqings/LLaVA-RLHF-7b-v1.5-224",
    local_dir="LLaVA-RLHF-7b-v1.5-224",
    repo_type="model"
)

sft_model_path  = "LLaVA-RLHF-7b-v1.5-224/sft_model"
lora_model_path = "LLaVA-RLHF-7b-v1.5-224/rlhf_lora_adapter_model"

# Tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(sft_model_path, use_fast=False)
print('tokenizer loaded')

# Load the base model (SFT)
model = LlavaLlamaForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16,   
    device_map="cuda:0",
    attn_implementation="eager",
    output_attentions=True
)
print('base model loaded')

# Apply LoRA weights
print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(model, lora_model_path)
print('Merging LoRA weights...')
model = model.merge_and_unload()

# Load the vision tower
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()

image_processor = vision_tower.image_processor

# Create the conversation and add a user message with image
conv = conv_templates['llava_v1'].copy()
raw_image = Image.open("images/img_prompt_1.jpg").convert("RGB")
DEFAULT_IMAGE_TOKEN = "<image>"
inp = DEFAULT_IMAGE_TOKEN + '\n' + "what is the name of the school mentioned in the image?"
conv.append_message(conv.roles[1], None)

# Get the formatted prompt
prompt = conv.get_prompt()
print("Prompt:\n", prompt)
image_tensor = process_images([raw_image], image_processor, model.config)

tokenizer.add_tokens(["<image>"], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
input_ids = tokenizer_image_token(prompt, tokenizer, 32000, return_tensors='pt').unsqueeze(0).to(model.device)
ids = input_ids[0]
model.to("cuda:0")
image_tensor = image_tensor.to(model.dtype).to("cuda:0")
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        images=image_tensor,
        return_dict_in_generate=True,
        output_attentions=True,
        max_new_tokens=200
    )

resp = outputs.sequences[0]
print(tokenizer.decode(resp))
print(outputs.attentions[0][0].shape)
