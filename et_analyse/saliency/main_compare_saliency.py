    
import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
from utils.data_loader import ETDataLoader
import numpy as np
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator
from eyetrackpy.data_generator.utils.saliency_metrics import compute_cc, compute_kl, compute_nss, compute_auc, compute_sim
import pandas as pd

def compute_saliency_metrics(model_saliency, human_sal, human_fixations):
    """
    Compute metrics between model and human saliency maps
    
    Args:
        model_saliency: np.array - Model saliency map
        human_sal: np.array - Human saliency map
        human_fixations: np.array - Human fixations (x, y coordinates)
    
    Returns:
        dict - Metrics for this subject/trial
    """
    import torch
    
    # Ensure same dimensions
    if model_saliency.shape != human_sal.shape:
        # Resize model saliency to match human saliency
        from PIL import Image
        model_pil = Image.fromarray((model_saliency * 255).astype(np.uint8))
        model_resized = model_pil.resize((human_sal.shape[1], human_sal.shape[0]))
        model_sal = np.array(model_resized) / 255.0
    else:
        model_sal = model_saliency
    
    # Convert to torch tensors
    model_tensor = torch.from_numpy(model_sal).float()
    human_sal_tensor = torch.from_numpy(human_sal).float()
    
    # Create fixation map from fixation coordinates
    fixation_map = create_fixation_map(human_fixations, human_sal.shape)
    fixation_tensor = torch.from_numpy(fixation_map).float()
    
    # Compute metrics
    metrics = {
        'sim': compute_sim(model_tensor, human_sal_tensor).item(),
        'auc': compute_auc(model_tensor, fixation_tensor).item(),
        'nss': compute_nss(model_tensor, fixation_tensor).item(),
        'kld': compute_kl(model_tensor, human_sal_tensor).item(),
        'cc': compute_cc(model_tensor, human_sal_tensor).item()
    }
    
    return metrics

def create_fixation_map(fixations, image_shape):
    """
    Create binary fixation map from fixation coordinates
    
    Args:
        fixations: np.array - Array of (x, y) fixation coordinates (0-1 range)
        image_shape: tuple - (height, width) of the image
    
    Returns:
        np.array - Binary fixation map
    """
    height, width = image_shape
    fixation_map = np.zeros((height, width), dtype=np.float32)
    
    if len(fixations) > 0:
        # Convert normalized coordinates to pixel coordinates
        x_coords = (fixations['x'] * width).astype(int)
        y_coords = (fixations['y'] * height).astype(int)
        
        # Ensure coordinates are within bounds
        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)
        
        # Set fixation points to 1
        fixation_map[y_coords, x_coords] = 1.0
    
    return fixation_map

if __name__ == "__main__":
    subjects = [1,2,3,4,5,6,7,8,9,10]
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et"
    save_path= cwd + '/results/et/'
    #load saliency
    subject_saliency = ETDataLoader().load_subject_saliency(subjects = subjects, raw_data_path = raw_data_path, processed_data_path = process_data_path)
    #load fixations
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    subject_fixations = ETDataLoader().load_subject_fixations(subjects = subjects, raw_data_path=raw_data_path)
    saliency_generator = SaliencyGenerator()

    results_path_saliency = save_path + '/visalformer/saliency'
    metrics_subjects = {'sim' : [], 'auc' : [], 'nss' : [], 'kld' : [], 'cc' : [], "SC" : [], "SE" : []}
    metrics_models = {"SC" : [], "SE" : []}
    for prompt_number, _ in prompts.items():
        box_image = ETDataLoader().find_image_in_screenshot(prompts_screenshots[prompt_number], images[prompt_number], draw_result=False)
        saliency_map_visalformer = np.load(results_path_saliency + '/saliency_trial_{}.npy'.format(prompt_number))
        saliency_coverage_visalformer = saliency_generator.compute_saliency_coverage(saliency_map_visalformer)
        saliency_entropy_visalformer = saliency_generator.compute_shannon_entropy(saliency_map_visalformer)
        metrics_models['SC'].append(saliency_coverage_visalformer)
        metrics_models['SE'].append(saliency_entropy_visalformer)
        
        for subject, saliency_maps in subject_saliency.items():
            fixations_trial = subject_fixations[subject]
            if prompt_number not in saliency_maps:
                continue
            saliency_map_trial = saliency_maps[prompt_number]
            fixations_trial = ETDataLoader().filter_rescale_fixations(fixations_trial, prompt_number, box_image)
            metrics = compute_saliency_metrics(saliency_map_visalformer, saliency_map_trial, fixations_trial)
            saliency_coverage = saliency_generator.compute_saliency_coverage(saliency_map_trial)
            saliency_entropy = saliency_generator.compute_shannon_entropy(saliency_map_trial)
            metrics_subjects['sim'].append(metrics['sim'])
            metrics_subjects['auc'].append(metrics['auc'])
            metrics_subjects['nss'].append(metrics['nss'])
            metrics_subjects['kld'].append(metrics['kld'])
            metrics_subjects['cc'].append(metrics['cc'])
            metrics_subjects['SC'].append(saliency_coverage)
            metrics_subjects['SE'].append(saliency_entropy)

    # Create Table 1: Comparison Metrics (SIM, AUC, NSS, KLD, CC)
    import pandas as pd
    comparison_metrics = {}
    for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
        comparison_metrics[f'{metric}_mean'] = np.mean(metrics_subjects[metric])
        comparison_metrics[f'{metric}_std'] = np.std(metrics_subjects[metric])
    
    comparison_df = pd.DataFrame([comparison_metrics])
    comparison_df.to_csv(os.path.join(results_path_saliency, 'comparison_metrics.csv'), index=False)
    
    # Create Table 2: Descriptive Statistics (SC, SE) for Model vs Subjects
    descriptive_metrics = {
        'model_SC_mean': np.mean(metrics_models['SC']),
        'model_SC_std': np.std(metrics_models['SC']),
        'model_SE_mean': np.mean(metrics_models['SE']),
        'model_SE_std': np.std(metrics_models['SE']),
        'subjects_SC_mean': np.mean(metrics_subjects['SC']),
        'subjects_SC_std': np.std(metrics_subjects['SC']),
        'subjects_SE_mean': np.mean(metrics_subjects['SE']),
        'subjects_SE_std': np.std(metrics_subjects['SE'])
    }
    
    descriptive_df = pd.DataFrame([descriptive_metrics])
    descriptive_df.to_csv(os.path.join(results_path_saliency, 'descriptive_metrics.csv'), index=False)
    
    # Print results
    print("Comparison Metrics (Model vs Human Saliency):")
    for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
        mean_val = comparison_metrics[f'{metric}_mean']
        std_val = comparison_metrics[f'{metric}_std']
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nDescriptive Statistics:")
    print(f"Model SC: {descriptive_metrics['model_SC_mean']:.4f} ± {descriptive_metrics['model_SC_std']:.4f}")
    print(f"Subjects SC: {descriptive_metrics['subjects_SC_mean']:.4f} ± {descriptive_metrics['subjects_SC_std']:.4f}")
    print(f"Model SE: {descriptive_metrics['model_SE_mean']:.4f} ± {descriptive_metrics['model_SE_std']:.4f}")
    print(f"Subjects SE: {descriptive_metrics['subjects_SE_mean']:.4f} ± {descriptive_metrics['subjects_SE_std']:.4f}")

