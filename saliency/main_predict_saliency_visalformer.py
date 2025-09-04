from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path
from eyetrackpy.data_generator.visalformer.saliency_predictor import VisalformerSaliencyPredictor
from eyetrackpy.data_generator.visalformer.dataset import DatasetLoader
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)

from utils.data_loader import ETDataLoader

def evaluation(batch_size):
    #load model
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = cwd + "/data/raw/et"
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path = raw_data_path)
    data_for_loader  =[]
    for prompt_number, prompt_text in prompts.items():
        data_for_loader.append((str(prompt_number), images[prompt_number], prompt_text))
    dataset_loader = DatasetLoader().create_dataloader(data_for_loader, batch_size=batch_size)
    
    model = VisalformerSaliencyPredictor()
    model.predict(dataset_loader, save_path=cwd + '/results/et/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = vars(parser.parse_args())

    evaluation(batch_size = args['batch_size'])
