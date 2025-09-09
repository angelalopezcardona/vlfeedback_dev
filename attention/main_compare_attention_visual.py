    
import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
from utils.data_loader import ETDataLoader
import numpy as np
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator
from eyetrackpy.data_generator.utils.saliency_metrics import compute_cc, compute_kl, compute_nss, compute_auc, compute_sim
import pandas as pd
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)

 # Define model names and their corresponding data
model_names = ['llava7', 'llava13']

    
  

def compute_table_metrics(metrics_subjects, save_path, print_table=True):
    # Generate LaTeX Table 1: Comparison Metrics
    def generate_latex_table1(metrics_subjects):
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Comparison Metrics Between Models and Human Saliency Maps}\n"
        latex_table += "\\label{tab:comparison_metrics}\n"
        latex_table += "\\begin{tabular}{l|ccccc}\n"
        latex_table += "\\hline\n"
        latex_table += "Model & SIM & AUC & NSS & KLD & CC \\\\\n"
        latex_table += "\\hline\n"
        
        for i, model_name in enumerate(model_names):
            
            model_metrics = metrics_subjects[model_name]

            row = f"{model_name}"
            for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
                mean_val = np.mean(model_metrics[metric])
                std_val = np.std(model_metrics[metric])
                row += f" & {mean_val:.3f} $\\pm$ {std_val:.3f}"
            row += " \\\\\n"
            latex_table += row
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        return latex_table
   
    # Create comparison metrics table
    comparison_data = []
    for i, model_name in enumerate(model_names):
        model_metrics = metrics_subjects[model_name]

        
        row_data = {'Model': model_name}
        for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
            mean_val = np.mean(model_metrics[metric])
            std_val = np.std(model_metrics[metric])
            row_data[f'{metric.upper()}_mean'] = mean_val
            row_data[f'{metric.upper()}_std'] = std_val
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(save_path, 'comparison_metrics_4models.csv'), index=False)
    if print_table:
        latex_table = generate_latex_table1(metrics_subjects)
        print(latex_table)
    return True

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
    subjects = [1,2,3,4,5,6,7,8,9,10,11,12,14,15]
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et"
    results_path_saliency = cwd + '/results/et/attention_rollout/'
    results_path_saliency_llava7 = results_path_saliency + 'llava-1.5-7b-hf/saliency/'
    results_path_saliency_llava13 = results_path_saliency + 'llava-1.5-13b-hf/saliency/'
    save_path = results_path_saliency + 'comparison/'
    os.makedirs(save_path, exist_ok=True)
    #load saliency
    subject_saliency = ETDataLoader().load_subject_saliency(subjects = subjects, raw_data_path = raw_data_path, processed_data_path = process_data_path)
    #load fixations
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    subject_fixations = ETDataLoader().load_subject_fixations(subjects = subjects, raw_data_path=raw_data_path)
    saliency_generator = SaliencyGenerator()

    
    
    metrics_subjects = {
        "llava7" : {'sim' : [], 'auc' : [], 'nss' : [], 'kld' : [], 'cc' : []},
        "llava13" : {'sim' : [], 'auc' : [], 'nss' : [], 'kld' : [], 'cc' : []},
       
    }
      
    for prompt_number, _ in prompts.items():
        box_image = ETDataLoader().find_image_in_screenshot(prompts_screenshots[prompt_number], images[prompt_number], draw_result=False)
        saliency_map_llava7 = np.load(results_path_saliency_llava7 + 'trial_{}/saliency_trial_{}_layer_gen_rollout.npy'.format(prompt_number, prompt_number))
        saliency_map_llava13 = np.load(results_path_saliency_llava13 + 'trial_{}/saliency_trial_{}_layer_gen_rollout.npy'.format(prompt_number, prompt_number))

        for subject, saliency_maps in subject_saliency.items():
            fixations_trial = subject_fixations[subject]
            if prompt_number not in saliency_maps:
                continue
            saliency_map_trial = saliency_maps[prompt_number]
            fixations_trial = ETDataLoader().filter_rescale_fixations(fixations_trial, prompt_number, box_image)
            
            saliency_coverage = saliency_generator.compute_saliency_coverage(saliency_map_trial)
            saliency_entropy = saliency_generator.compute_shannon_entropy(saliency_map_trial)

            metrics = compute_saliency_metrics(saliency_map_llava7, saliency_map_trial, fixations_trial)
            metrics_subjects["llava7"]['sim'].append(metrics['sim'])
            metrics_subjects["llava7"]["auc"].append(metrics['auc'])
            metrics_subjects["llava7"]["nss"].append(metrics['nss'])
            metrics_subjects["llava7"]["kld"].append(metrics['kld'])
            metrics_subjects["llava7"]["cc"].append(metrics['cc'])
            metrics = compute_saliency_metrics(saliency_map_llava13, saliency_map_trial, fixations_trial)
            metrics_subjects["llava13"]['sim'].append(metrics['sim'])
            metrics_subjects["llava13"]["auc"].append(metrics['auc'])
            metrics_subjects["llava13"]["nss"].append(metrics['nss'])
            metrics_subjects["llava13"]["kld"].append(metrics['kld'])
            metrics_subjects["llava13"]["cc"].append(metrics['cc'])



    
    compute_table_metrics(metrics_subjects, save_path=save_path, print_table=True)
    
    
   
    

    
    # # Print summary results
    # print("\n" + "="*80)
    # print("SUMMARY RESULTS")
    # print("="*80)
    # print("Comparison Metrics (Model vs Human Saliency):")
    # for i, (model_name, (main_key, sub_key)) in enumerate(zip(model_names, model_data_keys)):
    #     if sub_key is None:
    #         model_metrics = metrics_subjects[main_key]
    #     else:
    #         model_metrics = metrics_subjects[main_key][sub_key]
        
    #     print(f"\n{model_name}:")
    #     for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
    #         mean_val = np.mean(model_metrics[metric])
    #         std_val = np.std(model_metrics[metric])
    #         print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # print("\nDescriptive Statistics:")
    # print(f"Human SC: {human_sc_mean:.4f} ± {human_sc_std:.4f}")
    # print(f"Human SE: {human_se_mean:.4f} ± {human_se_std:.4f}")
    # for model_name, (main_key, sub_key) in zip(model_names, model_data_keys):
    #     if sub_key is None:
    #         model_metrics = metrics_saliency[main_key]
    #     else:
    #         model_metrics = metrics_saliency[main_key][sub_key]
        
    #     sc_mean = np.mean(model_metrics["SC"])
    #     sc_std = np.std(model_metrics["SC"])
    #     se_mean = np.mean(model_metrics["SE"])
    #     se_std = np.std(model_metrics["SE"])
    #     print(f"{model_name} SC: {sc_mean:.4f} ± {sc_std:.4f}")
    #     print(f"{model_name} SE: {se_mean:.4f} ± {se_std:.4f}")

