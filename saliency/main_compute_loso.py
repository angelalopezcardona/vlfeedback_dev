import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
from utils.data_loader import ETDataLoader
import numpy as np
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator
from eyetrackpy.data_generator.utils.saliency_metrics import compute_cc, compute_kl, compute_nss, compute_auc, compute_sim
import pandas as pd
import torch
from PIL import Image
import cv2

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
    
    # Ensure same dimensions
    if model_saliency.shape != human_sal.shape:
        # Resize model saliency to match human saliency
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
    
    if model_tensor.dim() == 2:
        model_tensor = model_tensor.unsqueeze(0).unsqueeze(0)   # (H,W) -> (1,1,H,W)
    elif model_tensor.dim() == 3:
        model_tensor = model_tensor.unsqueeze(1)                 # (B,H,W) -> (B,1,H,W)

    if human_sal_tensor.dim() == 2:
        human_sal_tensor = human_sal_tensor.unsqueeze(0).unsqueeze(0)
    elif human_sal_tensor.dim() == 3:
        human_sal_tensor = human_sal_tensor.unsqueeze(1)
        
    if fixation_tensor.dim() == 2:
        fixation_tensor = fixation_tensor.unsqueeze(0).unsqueeze(0)
    elif fixation_tensor.dim() == 3:
        fixation_tensor = fixation_tensor.unsqueeze(1)
        
    auc = compute_auc(model_tensor, fixation_tensor)
    auc = auc if isinstance(auc, torch.Tensor) else torch.tensor(auc, dtype=torch.float32)

    # Compute metrics
    metrics = {
        'sim': compute_sim(model_tensor, human_sal_tensor).item(),
        'auc': auc.item(),
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

def compute_average_saliency(subject_saliency, exclude_subject, trial_number):
    """
    Compute average saliency map for all subjects except the excluded one
    
    Args:
        subject_saliency: dict - Dictionary with subject saliency maps
        exclude_subject: int - Subject ID to exclude
        trial_number: int - Trial number
    
    Returns:
        np.array - Average saliency map
    """
    saliency_maps = []
    
    for subject_id, saliency_data in subject_saliency.items():
        if subject_id != exclude_subject and trial_number in saliency_data:
            saliency_maps.append(saliency_data[trial_number])
    
    if len(saliency_maps) == 0:
        return None
    
    # Average all saliency maps
    average_saliency = np.mean(saliency_maps, axis=0)
    return average_saliency

def compute_loso_metrics_for_trial(subject_saliency, subject_fixations, trial_number, box_image, 
                                  model_saliency_maps=None):
    """
    Compute LOSO metrics for a single trial
    
    Args:
        subject_saliency: dict - Dictionary with subject saliency maps
        subject_fixations: dict - Dictionary with subject fixation data
        trial_number: int - Trial number
        box_image: tuple - Bounding box coordinates
        model_saliency_maps: dict - Dictionary with model saliency maps for this trial
    
    Returns:
        dict - LOSO metrics for this trial (human ceiling + model comparisons)
    """
    subjects = list(subject_saliency.keys())
    
    # Initialize metrics storage
    trial_metrics = {
        'human_ceiling': {
            'sim': [],
            'auc': [],
            'nss': [],
            'kld': [],
            'cc': []
        },
        'models': {
            'visalformer': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []},
            'mdsem_500': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []},
            'mdsem_3000': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []},
            'mdsem_5000': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []}
        }
    }
    
    # For each subject, compute metrics against average of others (Human Ceiling)
    for test_subject in subjects:
        # Skip if test subject doesn't have data for this trial
        if trial_number not in subject_saliency[test_subject]:
            continue
            
        # Compute average saliency of all other subjects
        avg_saliency = compute_average_saliency(subject_saliency, test_subject, trial_number)
        if avg_saliency is None:
            continue
            
        # Get test subject's saliency and fixations
        test_saliency = subject_saliency[test_subject][trial_number]
        test_fixations = ETDataLoader().filter_rescale_fixations(
            subject_fixations[test_subject], trial_number, box_image
        )
        
        # Compute human ceiling metrics (averaged human vs left-out human)
        metrics = compute_saliency_metrics(avg_saliency, test_saliency, test_fixations)
        
        # Store human ceiling metrics
        for metric_name, metric_value in metrics.items():
            trial_metrics['human_ceiling'][metric_name].append(metric_value)
        
        # Compute model vs averaged human metrics (if model data is available)
        if model_saliency_maps is not None:
            for model_name, model_saliency in model_saliency_maps.items():
                if model_saliency is not None:
                    # Compare model against the averaged human saliency (not the left-out subject)
                    model_metrics = compute_saliency_metrics(model_saliency, avg_saliency, test_fixations)
                    
                    # Store model metrics
                    for metric_name, metric_value in model_metrics.items():
                        trial_metrics['models'][model_name][metric_name].append(metric_value)
    
    return trial_metrics

def load_model_saliency_maps(trial_number, results_path_saliency, results_path_saliency_mdsem, human_saliency_shape=None):
    """
    Load model saliency maps for a specific trial
    
    Args:
        trial_number: int - Trial number
        results_path_saliency: str - Path to Visalformer saliency maps
        results_path_saliency_mdsem: str - Path to MDSEM saliency maps
        human_saliency_shape: tuple - (height, width) of human saliency maps for resizing
    
    Returns:
        dict - Dictionary with model saliency maps
    """
    model_maps = {}
    
    try:
        # Load Visalformer
        visalformer_path = os.path.join(results_path_saliency, f'saliency_trial_{trial_number}.npy')
        if os.path.exists(visalformer_path):
            visalformer_saliency = np.load(visalformer_path)
            
            # Resize and normalize Visalformer if human saliency shape is provided
            if human_saliency_shape is not None:
                img_height, img_width = human_saliency_shape
                
                # Convert to PIL Image for resizing
                visalformer_pil = Image.fromarray(visalformer_saliency)
                visalformer_resized = visalformer_pil.resize((img_width, img_height), Image.BILINEAR)
                visalformer_saliency = np.array(visalformer_resized)
                
                # Normalize using cv2
                visalformer_saliency = cv2.normalize(visalformer_saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # Convert back to float and normalize to 0-1 range
                visalformer_saliency = visalformer_saliency.astype(np.float32) / 255.0
            
            model_maps['visalformer'] = visalformer_saliency
        else:
            model_maps['visalformer'] = None
            
        # Load MDSEM variants
        for variant in ['500', '3000', '5000']:
            mdsem_path = os.path.join(results_path_saliency_mdsem, f'img_prompt_{trial_number}_{variant}.npy')
            if os.path.exists(mdsem_path):
                model_maps[f'mdsem_{variant}'] = np.load(mdsem_path)
            else:
                model_maps[f'mdsem_{variant}'] = None
                
    except Exception as e:
        print(f"Warning: Could not load model saliency maps for trial {trial_number}: {e}")
        model_maps = {
            'visalformer': None,
            'mdsem_500': None,
            'mdsem_3000': None,
            'mdsem_5000': None
        }
    
    return model_maps

def compute_human_ceiling_loso(subject_saliency, subject_fixations, prompts, prompts_screenshots, images,
                              results_path_saliency, results_path_saliency_mdsem):
    """
    Compute human ceiling and model performance using Leave-One-Subject-Out cross-validation
    
    Args:
        subject_saliency: dict - Dictionary with subject saliency maps
        subject_fixations: dict - Dictionary with subject fixation data
        prompts: dict - Dictionary with prompts
        prompts_screenshots: dict - Dictionary with prompt screenshots
        images: dict - Dictionary with images
        results_path_saliency: str - Path to Visalformer saliency maps
        results_path_saliency_mdsem: str - Path to MDSEM saliency maps
    
    Returns:
        dict - Human ceiling and model performance metrics
    """
    print("Computing Human Ceiling and Model Performance using Leave-One-Subject-Out cross-validation...")
    
    # Initialize results storage
    all_trial_metrics = {
        'human_ceiling': {
            'sim': [],
            'auc': [],
            'nss': [],
            'kld': [],
            'cc': []
        },
        'models': {
            'visalformer': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []},
            'mdsem_500': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []},
            'mdsem_3000': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []},
            'mdsem_5000': {'sim': [], 'auc': [], 'nss': [], 'kld': [], 'cc': []}
        }
    }
    
    # Process each trial
    for trial_number in range(1, 31):  # Trials 1 to 30
        print(f"Processing trial {trial_number}/30...")
        
        # Find image in screenshot to get bounding box
        try:
            box_image = ETDataLoader().find_image_in_screenshot(
                prompts_screenshots[trial_number], 
                images[trial_number], 
                draw_result=False
            )
        except Exception as e:
            print(f"Warning: Could not process trial {trial_number}: {e}")
            continue
        
        # Get human saliency shape for resizing (use first available subject)
        human_saliency_shape = None
        for subject_id, saliency_data in subject_saliency.items():
            if trial_number in saliency_data:
                human_saliency_shape = saliency_data[trial_number].shape
                break
        
        # Load model saliency maps for this trial
        model_saliency_maps = load_model_saliency_maps(
            trial_number, results_path_saliency, results_path_saliency_mdsem, human_saliency_shape
        )
        
        # Compute LOSO metrics for this trial
        trial_metrics = compute_loso_metrics_for_trial(
            subject_saliency, subject_fixations, trial_number, box_image, model_saliency_maps
        )
        
        # Store trial results
        for category in ['human_ceiling', 'models']:
            for metric_name in ['sim', 'auc', 'nss', 'kld', 'cc']:
                if category == 'human_ceiling':
                    metric_values = trial_metrics[category][metric_name]
                    if len(metric_values) > 0:
                        all_trial_metrics[category][metric_name].append(np.mean(metric_values))
                    else:
                        print(f"Warning: No valid human ceiling metrics for trial {trial_number}, metric {metric_name}")
                else:  # models
                    for model_name in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000']:
                        metric_values = trial_metrics[category][model_name][metric_name]
                        if len(metric_values) > 0:
                            all_trial_metrics[category][model_name][metric_name].append(np.mean(metric_values))
                        else:
                            print(f"Warning: No valid {model_name} metrics for trial {trial_number}, metric {metric_name}")
    
    # Compute final results (mean and std across all trials)
    results = {}
    
    # Human ceiling results
    results['human_ceiling'] = {}
    for metric_name in ['sim', 'auc', 'nss', 'kld', 'cc']:
        trial_values = all_trial_metrics['human_ceiling'][metric_name]
        if len(trial_values) > 0:
            results['human_ceiling'][f'{metric_name}_mean'] = np.mean(trial_values)
            results['human_ceiling'][f'{metric_name}_std'] = np.std(trial_values)
            results['human_ceiling'][f'{metric_name}_values'] = trial_values
        else:
            results['human_ceiling'][f'{metric_name}_mean'] = np.nan
            results['human_ceiling'][f'{metric_name}_std'] = np.nan
            results['human_ceiling'][f'{metric_name}_values'] = []
    
    # Model results
    results['models'] = {}
    for model_name in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000']:
        results['models'][model_name] = {}
        for metric_name in ['sim', 'auc', 'nss', 'kld', 'cc']:
            trial_values = all_trial_metrics['models'][model_name][metric_name]
            if len(trial_values) > 0:
                results['models'][model_name][f'{metric_name}_mean'] = np.mean(trial_values)
                results['models'][model_name][f'{metric_name}_std'] = np.std(trial_values)
                results['models'][model_name][f'{metric_name}_values'] = trial_values
            else:
                results['models'][model_name][f'{metric_name}_mean'] = np.nan
                results['models'][model_name][f'{metric_name}_std'] = np.nan
                results['models'][model_name][f'{metric_name}_values'] = []
    
    return results

def save_loso_results(results, output_path):
    """
    Save LOSO results to CSV files
    
    Args:
        results: dict - Human ceiling and model performance metrics
        output_path: str - Path to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save human ceiling summary
    human_ceiling_data = []
    for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
        human_ceiling_data.append({
            'Metric': metric.upper(),
            'Mean': results['human_ceiling'][f'{metric}_mean'],
            'Std': results['human_ceiling'][f'{metric}_std'],
            'N_Trials': len(results['human_ceiling'][f'{metric}_values'])
        })
    
    human_ceiling_df = pd.DataFrame(human_ceiling_data)
    human_ceiling_df.to_csv(os.path.join(output_path, 'human_ceiling_loso_summary.csv'), index=False)
    
    # Save model performance summary
    model_data = []
    for model_name in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000']:
        for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
            model_data.append({
                'Model': model_name,
                'Metric': metric.upper(),
                'Mean': results['models'][model_name][f'{metric}_mean'],
                'Std': results['models'][model_name][f'{metric}_std'],
                'N_Trials': len(results['models'][model_name][f'{metric}_values'])
            })
    
    model_df = pd.DataFrame(model_data)
    model_df.to_csv(os.path.join(output_path, 'model_performance_loso_summary.csv'), index=False)
    
    # Save combined comparison table
    comparison_data = []
    for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
        row = {'Metric': metric.upper()}
        
        # Human ceiling
        row['Human_Ceiling'] = f"{results['human_ceiling'][f'{metric}_mean']:.4f} ± {results['human_ceiling'][f'{metric}_std']:.4f}"
        
        # Models
        for model_name in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000']:
            mean_val = results['models'][model_name][f'{metric}_mean']
            std_val = results['models'][model_name][f'{metric}_std']
            row[model_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_path, 'loso_comparison_table.csv'), index=False)
    
    # Save detailed results (individual trial values)
    detailed_data = []
    max_trials = max(
        max(len(results['human_ceiling'][f'{metric}_values']) for metric in ['sim', 'auc', 'nss', 'kld', 'cc']),
        max(len(results['models'][model][f'{metric}_values']) for model in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000'] for metric in ['sim', 'auc', 'nss', 'kld', 'cc'])
    )
    
    for trial_idx in range(max_trials):
        row = {'Trial': trial_idx + 1}
        
        # Human ceiling values
        for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
            values = results['human_ceiling'][f'{metric}_values']
            if trial_idx < len(values):
                row[f'Human_{metric.upper()}'] = values[trial_idx]
            else:
                row[f'Human_{metric.upper()}'] = np.nan
        
        # Model values
        for model_name in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000']:
            for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
                values = results['models'][model_name][f'{metric}_values']
                if trial_idx < len(values):
                    row[f'{model_name}_{metric.upper()}'] = values[trial_idx]
                else:
                    row[f'{model_name}_{metric.upper()}'] = np.nan
        
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(output_path, 'loso_detailed_results.csv'), index=False)
    
    print(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("="*80)
    
    print("\nHUMAN CEILING (Averaged Human vs Left-out Human):")
    print("-" * 50)
    for _, row in human_ceiling_df.iterrows():
        print(f"{row['Metric']}: {row['Mean']:.4f} ± {row['Std']:.4f} (N={row['N_Trials']})")
    
    print("\nMODEL PERFORMANCE (Model vs Averaged Human):")
    print("-" * 50)
    for model_name in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000']:
        print(f"\n{model_name.upper()}:")
        model_subset = model_df[model_df['Model'] == model_name]
        for _, row in model_subset.iterrows():
            print(f"  {row['Metric']}: {row['Mean']:.4f} ± {row['Std']:.4f} (N={row['N_Trials']})")
    
    print("\nPERFORMANCE GAPS (Human Ceiling - Model Performance):")
    print("-" * 50)
    for metric in ['sim', 'auc', 'nss', 'kld', 'cc']:
        human_mean = results['human_ceiling'][f'{metric}_mean']
        print(f"\n{metric.upper()}:")
        for model_name in ['visalformer', 'mdsem_500', 'mdsem_3000', 'mdsem_5000']:
            model_mean = results['models'][model_name][f'{metric}_mean']
            gap = human_mean - model_mean
            print(f"  {model_name}: {gap:.4f} (gap from human ceiling)")

if __name__ == "__main__":
    # Configuration
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et"
    save_path = cwd + '/results/et/'
    
    # Model saliency paths
    results_path_saliency = save_path + '/visalformer/saliency'
    results_path_saliency_mdsem = save_path + '/mdsem/salmaps_arrays'
    
    print("Loading data...")
    
    # Load data
    subject_saliency = ETDataLoader().load_subject_saliency(
        subjects=subjects, 
        raw_data_path=raw_data_path, 
        processed_data_path=process_data_path
    )
    
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    subject_fixations = ETDataLoader().load_subject_fixations(subjects=subjects, raw_data_path=raw_data_path)
    
    print(f"Loaded data for {len(subjects)} subjects")
    print(f"Processing {len(prompts)} trials")
    print(f"Model saliency paths:")
    print(f"  Visalformer: {results_path_saliency}")
    print(f"  MDSEM: {results_path_saliency_mdsem}")
    
    # Compute human ceiling and model performance using LOSO
    results = compute_human_ceiling_loso(
        subject_saliency, subject_fixations, prompts, prompts_screenshots, images,
        results_path_saliency, results_path_saliency_mdsem
    )
    
    # Save results
    output_path = os.path.join(save_path, 'loso_analysis')
    save_loso_results(results, output_path)
    
    print("\nLOSO analysis completed successfully!")
    print("This provides:")
    print("1. Human ceiling: Averaged human vs left-out human (theoretical upper bound)")
    print("2. Model performance: Each model vs averaged human saliency")
    print("3. Performance gaps: How far models are from human ceiling")
    print("Models are compared against the averaged human saliency for fair evaluation.")
