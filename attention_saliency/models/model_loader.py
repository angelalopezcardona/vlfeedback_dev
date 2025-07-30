
# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq






class ModelLoaderVisionSeq:
    def load_processor(self, model_name="llava-hf/llava-1.5-7b-hf"):
        processor = AutoProcessor.from_pretrained(model_name)
        return processor

    def load_model(self, model_name="llava-hf/llava-1.5-7b-hf"):
        
        model = AutoModelForVision2Seq.from_pretrained(model_name, device_map="auto", trust_remote_code=True, output_attentions=True)
        return model






class ModelLoaderFactory:
    def get_model_loader(self, loader_type):
        if loader_type == "causalLM":
            return ModelLoaderVisionSeq()
        else:
            raise ValueError(f"Unknown model loader type: {loader_type}")
