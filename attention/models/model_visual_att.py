# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq
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

class ModelVisualAttentionExtractor(ModelAttentionExtractor):
    def __init__(self, model_name, model_type):
        super().__init__(model_name, model_type)
        self.model_name = model_name
        self.model_type = model_type
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, output_attentions=True)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract_attention(self, texts_trials: dict, word_level: bool = True):
        attention_trials = {}
        for trial, list_text in texts_trials.items():
            if not '.' in str(trial):
                continue
            print("trial", trial)
            list_word_original = [str(x) for x in list_text]
            text = " ".join(list_word_original)
            list_word_original = [x.lower() for x in list_word_original]
            input_ids = self.tokenize_text(self.tokenizer, text)
            attention = self.get_attention_model(self.model, input_ids)
            # try:
            attention_trials[trial] = self.process_attention(
                attention,
                input_ids,
                text=text,
                list_word_original=list_word_original,
                word_level=word_level,
            )
            # except Exception as e:
            #     print(trial, "error:", e)
        return attention_trials