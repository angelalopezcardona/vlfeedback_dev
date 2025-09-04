    
import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
from utils.data_loader import ETDataLoader
import numpy as np
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator
if __name__ == "__main__":
    subjects = [10]
    # paths
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et/saliency/"
    #load raw data
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    box_images = {}
    for prompt_number, _ in prompts.items():
        box_images[prompt_number] = ETDataLoader().find_image_in_screenshot(prompts_screenshots[prompt_number], images[prompt_number], draw_result=True, out_path=raw_data_path + "/match_vis_{}.png".format(str(prompt_number)))
    subject_fixations = ETDataLoader().load_subject_fixations(subjects=subjects, raw_data_path=raw_data_path)
    # generate and load saliency
    saliency_generator = SaliencyGenerator()
    for subject, data in subject_fixations.items():
        process_data_path_subject = process_data_path + "participant_" + str(subject) + "/"
        os.makedirs(process_data_path_subject, exist_ok=True)
        for prompt_number, _ in prompts.items():
            if '.' not in str(prompt_number):
                fixations_image = ETDataLoader().filter_rescale_fixations(data, prompt_number, box_images[prompt_number])
                fixations = data[data['USER']==str(prompt_number)][['x','y']]
                if fixations_image.empty:
                    continue
                saliency_map, overlay = saliency_generator.generate_saliency_map(prompts_screenshots[prompt_number], fixations, scale_fixations=True, sigma=60, alpha=0.6, weight_factor=3.0, return_overlay=True)
                saliency_generator.save_saliency_map(overlay, folder = process_data_path_subject, figure_name = "saliency_screenshot_{}.png".format(str(prompt_number)))
                saliency_map, overlay = saliency_generator.generate_saliency_map(images[prompt_number], fixations_image, scale_fixations=True, sigma=60, alpha=0.6, weight_factor=3.0, return_overlay=True)
                np.save(process_data_path_subject + "saliency_{}.npy".format(str(prompt_number)), saliency_map)
                saliency_generator.save_saliency_map(overlay, folder = process_data_path_subject, figure_name = "saliency_{}.png".format(str(prompt_number)))
               