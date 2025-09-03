    
import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
from utils.data_loader import ETDataLoader
import numpy as np
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator
from eyetrackpy.data_generator.utils.saliency_metrics import compute_cc, compute_kl, compute_nss, compute_auc, compute_sim
import pandas as pd

 # Define model names and their corresponding data
model_names = ['Visalformer', 'MDSEM-500', 'MDSEM-3000', 'MDSEM-5000']
model_data_keys = [
    ('visalformer', None),
    ('mdsem', '500'),
    ('mdsem', '3000'),
    ('mdsem', '5000')
]
    
def compute_descriptive_statistics(metrics_saliency, print_table=True):
     # Generate LaTeX Table 2: Descriptive Statistics
    def generate_latex_table2():
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Descriptive Statistics: Saliency Coverage (SC) and Shannon Entropy (SE)}\n"
        latex_table += "\\label{tab:descriptive_stats}\n"
        latex_table += "\\begin{tabular}{l|cc}\n"
        latex_table += "\\hline\n"
        latex_table += "Group & SC & SE \\\\\n"
        latex_table += "\\hline\n"
        
        # Human data
        latex_table += f"Human & {human_sc_mean:.3f} $\\pm$ {human_sc_std:.3f} & {human_se_mean:.3f} $\\pm$ {human_se_std:.3f} \\\\\n"
        
        # Model data
        for model_name, (main_key, sub_key) in zip(model_names, model_data_keys):
            if sub_key is None:
                model_metrics = metrics_saliency[main_key]
            else:
                model_metrics = metrics_saliency[main_key][sub_key]
            
            sc_mean = np.mean(model_metrics["SC"])
            sc_std = np.std(model_metrics["SC"])
            se_mean = np.mean(model_metrics["SE"])
            se_std = np.std(model_metrics["SE"])
            
            latex_table += f"{model_name} & {sc_mean:.3f} $\\pm$ {sc_std:.3f} & {se_mean:.3f} $\\pm$ {se_std:.3f} \\\\\n"
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        return latex_table
    # Create Table 2: Descriptive Statistics (SC, SE) for 5 groups (Human + 4 models)
    descriptive_data = []
    
    # Add human data
    human_sc_mean = np.mean(metrics_saliency["subject"]["SC"])
    human_sc_std = np.std(metrics_saliency["subject"]["SC"])
    human_se_mean = np.mean(metrics_saliency["subject"]["SE"])
    human_se_std = np.std(metrics_saliency["subject"]["SE"])
    descriptive_data.append({
        'Group': 'Human',
        'SC_mean': human_sc_mean,
        'SC_std': human_sc_std,
        'SE_mean': human_se_mean,
        'SE_std': human_se_std
    })
    
    # Add model data
    for model_name, (main_key, sub_key) in zip(model_names, model_data_keys):
        if sub_key is None:
            model_metrics = metrics_saliency[main_key]
        else:
            model_metrics = metrics_saliency[main_key][sub_key]
        
        sc_mean = np.mean(model_metrics["SC"])
        sc_std = np.std(model_metrics["SC"])
        se_mean = np.mean(model_metrics["SE"])
        se_std = np.std(model_metrics["SE"])
        
        descriptive_data.append({
            'Group': model_name,
            'SC_mean': sc_mean,
            'SC_std': sc_std,
            'SE_mean': se_mean,
            'SE_std': se_std
        })
    
    descriptive_df = pd.DataFrame(descriptive_data)
    descriptive_df.to_csv(os.path.join(results_path_saliency, 'descriptive_metrics_5groups.csv'), index=False)
    if print_table:
        latex_table = generate_latex_table2(metrics_saliency)
        print(latex_table)
    return True
    
    
   

def compute_table_metrics(metrics_subjects, print_table=True):
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
        
        for i, (model_name, (main_key, sub_key)) in enumerate(zip(model_names, model_data_keys)):
            if sub_key is None:
                model_metrics = metrics_subjects[main_key]
            else:
                model_metrics = metrics_subjects[main_key][sub_key]
            
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
    for i, (model_name, (main_key, sub_key)) in enumerate(zip(model_names, model_data_keys)):
        if sub_key is None:
            model_metrics = metrics_subjects[main_key]
        else:
            model_metrics = metrics_subjects[main_key][sub_key]
        
        row_data = {'Model': model_name}
        for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
            mean_val = np.mean(model_metrics[metric])
            std_val = np.std(model_metrics[metric])
            row_data[f'{metric.upper()}_mean'] = mean_val
            row_data[f'{metric.upper()}_std'] = std_val
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(results_path_saliency, 'comparison_metrics_4models.csv'), index=False)
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
    results_path_saliency_mdsem = save_path + '/mdsem/salmaps_arrays'
    metrics_subjects = {
        "visalformer" : {'sim' : [], 'auc' : [], 'nss' : [], 'kld' : [], 'cc' : []},
        "mdsem" : {
            "500" : {'sim' : [], 'auc' : [], 'nss' : [], 'kld' : [], 'cc' : []}, 
            "3000" : {'sim' : [], 'auc' : [], 'nss' : [], 'kld' : [], 'cc' : []}, 
            "5000" : {'sim' : [], 'auc' : [], 'nss' : [], 'kld' : [], 'cc' : []}
        }
    }
      
    metrics_saliency = {
        "subject" : {"SC" : [], "SE" : []}, 
        "visalformer" : {"SC": [], "SE" : []}, 
        "mdsem" : {"500" : {"SC" : [], "SE" : []}, '3000' : {"SC" : [], "SE" : []}, '5000' : {"SC" : [], "SE" : []}}}
    for prompt_number, _ in prompts.items():
        box_image = ETDataLoader().find_image_in_screenshot(prompts_screenshots[prompt_number], images[prompt_number], draw_result=False)
        saliency_map_visalformer = np.load(results_path_saliency + '/saliency_trial_{}.npy'.format(prompt_number))
        saliency_map_mdsem_05 = np.load(results_path_saliency_mdsem + '/img_prompt_{}_500.npy'.format(prompt_number))
        saliency_map_mdsem_3 = np.load(results_path_saliency_mdsem + '/img_prompt_{}_5000.npy'.format(prompt_number))
        saliency_map_mdsem_5 = np.load(results_path_saliency_mdsem + '/img_prompt_{}_5000.npy'.format(prompt_number))
        saliency_coverage_visalformer = saliency_generator.compute_saliency_coverage(saliency_map_visalformer)
        saliency_entropy_visalformer = saliency_generator.compute_shannon_entropy(saliency_map_visalformer)
        metrics_saliency["visalformer"]['SC'].append(saliency_coverage_visalformer)
        metrics_saliency["visalformer"]["SE"].append(saliency_entropy_visalformer)
        saliency_coverage_mdsem_05 = saliency_generator.compute_saliency_coverage(saliency_map_mdsem_05)
        saliency_entropy_mdsem_05 = saliency_generator.compute_shannon_entropy(saliency_map_mdsem_05)
        metrics_saliency["mdsem"]["500"]["SC"].append(saliency_coverage_mdsem_05)
        metrics_saliency["mdsem"]["500"]["SE"].append(saliency_entropy_mdsem_05)
        saliency_coverage_mdsem_3 = saliency_generator.compute_saliency_coverage(saliency_map_mdsem_3)
        saliency_entropy_mdsem_3 = saliency_generator.compute_shannon_entropy(saliency_map_mdsem_3)
        metrics_saliency["mdsem"]["3000"]["SC"].append(saliency_coverage_mdsem_3)
        metrics_saliency["mdsem"]["3000"]["SE"].append(saliency_entropy_mdsem_3)
        saliency_coverage_mdsem_5 = saliency_generator.compute_saliency_coverage(saliency_map_mdsem_5)
        saliency_entropy_mdsem_5 = saliency_generator.compute_shannon_entropy(saliency_map_mdsem_5)
        metrics_saliency["mdsem"]["5000"]["SC"].append(saliency_coverage_mdsem_5)
        metrics_saliency["mdsem"]["5000"]["SE"].append(saliency_entropy_mdsem_5)
        
        for subject, saliency_maps in subject_saliency.items():
            fixations_trial = subject_fixations[subject]
            if prompt_number not in saliency_maps:
                continue
            saliency_map_trial = saliency_maps[prompt_number]
            fixations_trial = ETDataLoader().filter_rescale_fixations(fixations_trial, prompt_number, box_image)
            
            saliency_coverage = saliency_generator.compute_saliency_coverage(saliency_map_trial)
            saliency_entropy = saliency_generator.compute_shannon_entropy(saliency_map_trial)
            metrics_saliency["subject"]["SC"].append(saliency_coverage)
            metrics_saliency["subject"]["SE"].append(saliency_entropy)
            metrics = compute_saliency_metrics(saliency_map_visalformer, saliency_map_trial, fixations_trial)
            metrics_subjects["visalformer"]['sim'].append(metrics['sim'])
            metrics_subjects["visalformer"]["auc"].append(metrics['auc'])
            metrics_subjects["visalformer"]["nss"].append(metrics['nss'])
            metrics_subjects["visalformer"]["kld"].append(metrics['kld'])
            metrics_subjects["visalformer"]["cc"].append(metrics['cc'])
            metrics = compute_saliency_metrics(saliency_map_mdsem_05, saliency_map_trial, fixations_trial)
            metrics_subjects["mdsem"]["500"]["sim"].append(metrics['sim'])
            metrics_subjects["mdsem"]["500"]["auc"].append(metrics['auc'])
            metrics_subjects["mdsem"]["500"]["nss"].append(metrics['nss'])
            metrics_subjects["mdsem"]["500"]["kld"].append(metrics['kld'])
            metrics_subjects["mdsem"]["500"]["cc"].append(metrics['cc'])
            metrics = compute_saliency_metrics(saliency_map_mdsem_3, saliency_map_trial, fixations_trial)
            metrics_subjects["mdsem"]["3000"]["sim"].append(metrics['sim'])
            metrics_subjects["mdsem"]["3000"]["auc"].append(metrics['auc'])
            metrics_subjects["mdsem"]["3000"]["nss"].append(metrics['nss'])
            metrics_subjects["mdsem"]["3000"]["kld"].append(metrics['kld'])
            metrics_subjects["mdsem"]["3000"]["cc"].append(metrics['cc'])
            metrics = compute_saliency_metrics(saliency_map_mdsem_5, saliency_map_trial, fixations_trial)
            metrics_subjects["mdsem"]["5000"]["sim"].append(metrics['sim'])
            metrics_subjects["mdsem"]["5000"]["auc"].append(metrics['auc'])
            metrics_subjects["mdsem"]["5000"]["nss"].append(metrics['nss'])
            metrics_subjects["mdsem"]["5000"]["kld"].append(metrics['kld'])
            metrics_subjects["mdsem"]["5000"]["cc"].append(metrics['cc'])


    
    compute_table_metrics(metrics_subjects, print_table=True)
    compute_descriptive_statistics(metrics_saliency, print_table=True)
    
    
   
    

    
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

