import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import ttest_rel, wilcoxon, shapiro
import statsmodels.formula.api as smf


class SaliencyEvaluator:
    def __init__(self, user_root="users", synthetic_salmaps_root="synthetic_salmaps", original_images_root="images", output_dir="salmaps", n_prompts=30):
        """
        A comprehensive saliency evaluation framework for comparing human eye-tracking data with synthetic saliency maps.
        
        This class provides a complete pipeline for saliency map analysis, including:
        
        CORE FUNCTIONALITY:
        - Processes user fixation data from multiple participants across different time windows (500ms, 3000ms, 5000ms, full)
        - Generates average human saliency maps by aggregating individual participant data
        - Compares human saliency maps with synthetic model predictions using three standard metrics:
          * CC (Pearson Correlation Coefficient): Linear correlation between maps
          * KL (Kullback-Leibler divergence): Information loss between probability distributions  
          * SIM (Similarity): Intersection of normalized distributions
        
        ADVANCED ANALYSIS METHODS:
        - evaluate_loso(): Leave-One-Subject-Out cross-validation for unbiased model evaluation
        - analyze_loso_results(): Comprehensive statistical analysis of LOSO results
        - analyze_human_by_suffix(): Detailed analysis of human performance across time windows
        - wilcoxon_loso_vs_global(): Statistical comparison between LOSO and global evaluation methods
        - _run_stat_tests(): Paired statistical tests (t-test/Wilcoxon) between human and model performance
        - _run_mixed_effects(): Linear mixed-effects models accounting for image and participant random effects
        - create_metric_boxplots(): Visualization of performance metrics across time windows
        
        OUTPUT STRUCTURE:
        Creates organized output directories:
        - salmaps/: Average human saliency maps (grayscale PNG)
        - salmaps_vis/: Overlay visualizations of saliency on original images
        - salmaps_arrays/: Raw saliency data (NumPy arrays)
        - metrics/: All statistical results and analysis files
        - plots/: Visualization outputs (boxplots, etc.)
        
        KEY OUTPUT FILES:
        - metrics_results.csv: Individual image-level metrics
        - global_metrics.csv: Aggregated metrics by time window
        - metrics_loso.csv: Leave-one-subject-out evaluation results
        - stat_tests_results.csv: Statistical test results (human vs model)
        - human_ceiling.csv: Human performance ceiling estimates
        - wilcoxon_loso_vs_global.csv: Comparison of evaluation methodologies
        
        Args:
            user_root: Root directory containing participant data (default: "users")
            synthetic_salmaps_root: Directory with synthetic saliency maps (default: "synthetic_salmaps")
            original_images_root: Directory with original stimulus images (default: "images")
            output_dir: Base output directory for all results (default: "salmaps")
            n_prompts: Number of image prompts to process (default: 30)
        """
        self.user_root = user_root
        self.synthetic_salmaps_root = synthetic_salmaps_root
        self.original_images_root = original_images_root
        self.output_dir = output_dir
        self.n_prompts = n_prompts
        self.suffixes = ['500', '3000', '5000', 'full']  # Available suffixes
        
        # Create output directories
        self.salmaps_dir = os.path.join(self.output_dir, 'salmaps')
        self.salmaps_vis_dir = os.path.join(self.output_dir, 'salmaps_vis')
        self.salmaps_arrays_dir = os.path.join(self.output_dir, 'salmaps_arrays')
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        
        os.makedirs(self.salmaps_dir, exist_ok=True)
        os.makedirs(self.salmaps_vis_dir, exist_ok=True)
        os.makedirs(self.salmaps_arrays_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Dictionary to store metrics for results
        self.metrics_tobii = self._create_metrics_dict()
        
        # Automatically start evaluation
        self._evaluate()
    
    @staticmethod
    def _create_metrics_dict():
        """Create an empty dictionary for metrics."""
        return {
            "CC": [], 
            "KL": [], 
            "SIM": [],
            "img_id": [],  # Store image ID for each metric
            "suffix": [],  # Store suffix for each metric
            "type": []     # Store map type for each metric
        }
    
    @staticmethod
    def cc(pred, gt):
        """Calculate Pearson correlation coefficient."""
        return pearsonr(pred.flatten(), gt.flatten())[0]

    @staticmethod
    def kl(pred, gt):
        """Calculate Kullback-Leibler divergence."""
        # Ensure positive values and normalize
        pred = np.maximum(pred, 1e-8)
        gt = np.maximum(gt, 1e-8)
        
        pred = pred / pred.sum()
        gt = gt / gt.sum()
        
        # Calculate KL divergence with safety checks
        ratio = gt / pred
        log_ratio = np.log(ratio)
        kl_div = np.sum(gt * log_ratio)
        
        # Check for invalid values
        if np.isnan(kl_div) or np.isinf(kl_div):
            return 0.0  # Return 0 for invalid cases
        
        return kl_div

    @staticmethod
    def sim(pred, gt):
        """Calculate similarity."""
        pred = pred / (pred.sum() + 1e-8)
        gt = gt / (gt.sum() + 1e-8)
        return np.sum(np.minimum(pred, gt))
    
    def _save_average_saliency_map(self, gt_raw, img_id, map_type, suffix, output_type='salmaps'):
        """
        Save the average saliency map for a specific trial and map type.
        
        Args:
            gt_raw: Raw average saliency map
            img_id: Image ID
            map_type: Map type ('salmaps', 'salmaps_vis', 'salmaps_array')
            suffix: Suffix of the synthetic map (500, 3000, 5000, 'avg')
            output_type: Type of output ('salmaps', 'salmaps_vis', 'salmaps_arrays')
        """
        if map_type == 'salmaps_array' or output_type == 'salmaps_arrays':
            # Save as numpy array
            output_path = os.path.join(self.salmaps_arrays_dir, f'img_prompt_{img_id}_{map_type}_{suffix}_avg.npy')
            np.save(output_path, gt_raw)
        else:
            # Save as image
            avg_img_255 = cv2.normalize(gt_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if map_type == 'salmaps_vis':
                output_path = os.path.join(self.salmaps_vis_dir, f'img_prompt_{img_id}_{map_type}_{suffix}_avg.png')
            else:
                output_path = os.path.join(self.salmaps_dir, f'img_prompt_{img_id}_{map_type}_{suffix}_avg.png')
            cv2.imwrite(output_path, avg_img_255)
    
    def _save_overlay(self, gt_raw, original_path, img_id, map_type, suffix):
        """
        Save the overlay of the average saliency map on the original image.
        
        Args:
            gt_raw: Raw average saliency map
            original_path: Path of the original image
            img_id: Image ID
            map_type: Map type ('salmaps', 'salmaps_vis', 'salmaps_array')
            suffix: Suffix of the synthetic map (500, 3000, 5000, 'avg')
        """
        avg_img_255 = cv2.normalize(gt_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(avg_img_255, cv2.COLORMAP_JET)

        if os.path.isfile(original_path):
            original = cv2.imread(original_path)
            original = cv2.resize(original, (heatmap.shape[1], heatmap.shape[0]))
            overlay = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)
            output_path = os.path.join(self.salmaps_vis_dir, f'img_prompt_{img_id}_{map_type}_{suffix}_overlay.png')
            cv2.imwrite(output_path, overlay)
        else:
            print(f"[Warning] Missing original image for img_prompt_{img_id}")
    
    def _compute_metrics(self, synthetic_norm, gt_norm, img_id, suffix, metrics_dict):
        """
        Calculate metrics between synthetic and ground truth maps.
        
        Args:
            synthetic_norm: Normalized synthetic saliency map
            gt_norm: Normalized ground truth saliency map
            img_id: Image ID
            suffix: Suffix of the synthetic map
            metrics_dict: Dictionary to store metrics
        
        Returns:
            dict: Calculated metrics for this image
        """
        # Debug checks for NaN values
        if np.isnan(synthetic_norm).any() or np.isnan(gt_norm).any():
            print(f"[Warning] NaN detected in maps for img_prompt_{img_id}_{suffix}")
            print(f"  Synthetic min/max: {synthetic_norm.min():.6f}/{synthetic_norm.max():.6f}")
            print(f"  GT min/max: {gt_norm.min():.6f}/{gt_norm.max():.6f}")
        
        metric_cc = self.cc(synthetic_norm, gt_norm)
        metric_kl = self.kl(synthetic_norm, gt_norm)
        metric_sim = self.sim(synthetic_norm, gt_norm)
        
        # Debug check for NaN in metrics
        if np.isnan(metric_kl):
            print(f"[Warning] KL is NaN for img_prompt_{img_id}_{suffix}")
            print(f"  CC: {metric_cc}, SIM: {metric_sim}")
        
        metrics_dict["img_id"].append(img_id)
        metrics_dict["suffix"].append(suffix)
        metrics_dict["CC"].append(metric_cc)
        metrics_dict["KL"].append(metric_kl)
        metrics_dict["SIM"].append(metric_sim)
        
        return {
            "CC": metric_cc,
            "KL": metric_kl,
            "SIM": metric_sim
        }
    
    
    
    def _collect_saliency_maps(self, img_id, suffix):
        """
        Collects the .npy maps of all users for an image and suffix.
        """
        saliency_stack = []

        for user in os.listdir(self.user_root):
            if user.startswith('participant_') and os.path.isdir(os.path.join(self.user_root, user)):
                # Skip users that contain 'sebastian' in their name
                if 'sebastian' in user.lower():
                    continue
                user_path0 = os.path.join(self.user_root, user)
                user_path = os.path.join(user_path0, 'session_1')
                
                # Look for any directory containing 'results_' in the name
                for sub_dir in os.listdir(user_path):
                    if 'results_' in sub_dir and os.path.isdir(os.path.join(user_path, sub_dir)):
                        saliency_path = os.path.join(
                            user_path, sub_dir, "salmaps_array",
                            f'img_prompt_{img_id}_saliency_array_{suffix}.npy'
                        )
                        
                        if os.path.isfile(saliency_path):
                            arr = np.load(saliency_path).astype(np.float32)
                            saliency_stack.append(arr)
                            break  # found for this user → move to next
                        else:
                            print(f"[Warning] Missing NPY for img_prompt_{img_id} suffix {suffix} user {user}")

        return saliency_stack



    
    def _load_synthetic_map(self, img_id, suffix, map_type='salmaps'):
        """
        Load synthetic saliency map from either salmaps or salmaps_arrays.
        
        Args:
            img_id: Image ID
            suffix: Suffix (500, 3000, 5000, full)
            map_type: Type of map ('salmaps' or 'salmaps_arrays')
        
        Returns:
            numpy.ndarray: Loaded synthetic map
        """
        # For 'full' suffix, use '5000' synthetic maps for comparison
        synthetic_suffix = '5000' if suffix == 'full' else suffix
        
        if map_type == 'salmaps':
            # Load image file
            synthetic_path = os.path.join(self.synthetic_salmaps_root, 'salmaps', f'img_prompt_{img_id}_{synthetic_suffix}.jpg')
            if os.path.isfile(synthetic_path):
                return cv2.imread(synthetic_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        elif map_type == 'salmaps_arrays':
            # Load numpy array
            synthetic_path = os.path.join(self.synthetic_salmaps_root, 'salmaps_arrays', f'img_prompt_{img_id}_{synthetic_suffix}.npy')
            if os.path.isfile(synthetic_path):
                return np.load(synthetic_path).astype(np.float32)
        
        return None
    
    def process_image(self, img_id):
        original_path = os.path.join(self.original_images_root, f'img_prompt_{img_id}.jpg')

        for suffix in self.suffixes:
            # 1️⃣ Load all NPY maps for this img_id + suffix
            saliency_stack = self._collect_saliency_maps(img_id, suffix)
            if not saliency_stack:
                print(f"[Skip] No NPY maps for img_prompt_{img_id} suffix {suffix}")
                continue

            # 2️⃣ Float average
            gt_raw = np.mean(np.stack(saliency_stack), axis=0)

            # 3️⃣ Save average NPY
            npy_path = os.path.join(self.salmaps_arrays_dir, f'img_prompt_{img_id}_salmaps_array_{suffix}_avg.npy')
            np.save(npy_path, gt_raw)

            # 4️⃣ Save grayscale (salmaps_avg)
            avg_img_255 = cv2.normalize(gt_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gray_path = os.path.join(self.salmaps_dir, f'img_prompt_{img_id}_salmaps_{suffix}_avg.png')
            cv2.imwrite(gray_path, avg_img_255)

            # 5️⃣ Create and save overlay (salmaps_vis_avg)
            if os.path.isfile(original_path):
                heatmap = cv2.applyColorMap(avg_img_255, cv2.COLORMAP_JET)
                original = cv2.imread(original_path)
                original = cv2.resize(original, (heatmap.shape[1], heatmap.shape[0]))
                alpha = getattr(self, "alpha", 0.6)
                overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
                vis_path = os.path.join(self.salmaps_vis_dir, f'img_prompt_{img_id}_salmaps_vis_{suffix}_avg.png')
                cv2.imwrite(vis_path, overlay)
            else:
                print(f"[Warning] Missing original image for img_prompt_{img_id}")
            
            # 6️⃣ Calculate metrics comparing synthetic map with average GT (NPY only)
            # For 'full' suffix, compare with '5000' of synthetics
            if suffix == 'full':
                synthetic_suffix = '5000'
            else:
                synthetic_suffix = suffix
                
            for synthetic_map_type in ['salmaps_arrays']:  # Only NPY files
                synthetic = self._load_synthetic_map(img_id, suffix, synthetic_map_type)
                if synthetic is None:
                    print(f"[Warning] Missing synthetic {synthetic_map_type} for img_prompt_{img_id}_{suffix} (using {synthetic_suffix})")
                    continue

                synthetic_resized = cv2.resize(synthetic, (gt_raw.shape[1], gt_raw.shape[0]), interpolation=cv2.INTER_LINEAR)

                metrics = self._compute_metrics(synthetic_resized, gt_raw, img_id, suffix, self.metrics_tobii)
                self.metrics_tobii["type"].append("salmaps_array")

                print(f"[Metrics img_prompt_{img_id}_{suffix} vs synthetic_{synthetic_suffix}] "
                    f"CC: {metrics['CC']:.4f}, KL: {metrics['KL']:.4f}, SIM: {metrics['SIM']:.4f}")

            



    def _evaluate(self):
        """
        Evaluate saliency maps for a set of images and save results in CSV.
        """
        for i in range(1, self.n_prompts + 1):
            self.process_image(i)
        
        # Create DataFrames for single image results
        df_results = pd.DataFrame(self.metrics_tobii)
        
        # Save DataFrames to CSV files in metrics directory
        df_results.to_csv(os.path.join(self.metrics_dir, 'metrics_results.csv'), index=False)
        
        # Calculate and save global metrics for each suffix and type
        global_metrics = []
        for suffix in self.suffixes:
            for map_type in ['salmaps_array']:  # Only NPY files
                # For 'full' suffix, use data from '5000' suffix for synthetics
                if suffix == 'full':
                    # Create a copy of '5000' suffix data for 'full' suffix
                    suffix_5000_data = self.metrics_tobii.copy()
                    # Replace all '5000' suffixes with 'full' for synthetics
                    for i, s in enumerate(suffix_5000_data["suffix"]):
                        if s == '5000':
                            suffix_5000_data["suffix"][i] = 'full'
                    global_metrics.append(self._calculate_global_metrics_for_suffix_and_type('full', map_type, suffix_5000_data))
                else:
                    global_metrics.append(self._calculate_global_metrics_for_suffix_and_type(suffix, map_type))
        
        # Create and save DataFrame with global metrics
        df_global_metrics = pd.DataFrame(global_metrics)
        df_global_metrics.to_csv(os.path.join(self.metrics_dir, 'global_metrics.csv'), index=False)
        
        print(f"\nCSV files have been saved in directory: {self.metrics_dir}")
    
    def _calculate_global_metrics_for_suffix_and_type(self, suffix, map_type, custom_data=None):
        """
        Calculate global metrics for a given suffix and map type.
        
        Args:
            suffix: Suffix to filter by
            map_type: Map type to filter by
            custom_data: Optional custom data to use instead of self.metrics_tobii
            
        Returns:
            dict: Dictionary containing global metrics for this suffix and type
        """
        global_metrics = {
            "suffix": suffix,
            "type": map_type,
        }
        
        # Use custom data if provided, otherwise use self.metrics_tobii
        data_to_use = custom_data if custom_data is not None else self.metrics_tobii
        
        # Filter metrics for this suffix and type
        suffix_mask = [s == suffix for s in data_to_use["suffix"]]
        type_mask = [t == map_type for t in data_to_use["type"]]
        combined_mask = [s and t for s, t in zip(suffix_mask, type_mask)]
        
        print(f"\n=== Final average metrics for {map_type} - {suffix} ===")
        for key in ["CC", "KL", "SIM"]:
            if data_to_use[key] and any(combined_mask):
                values = np.array(data_to_use[key])[combined_mask]
                mean_val = values.mean()
                std_val = values.std()
                
                # Save both mean and standard deviation
                global_metrics[f"{key}_mean"] = mean_val
                global_metrics[f"{key}_std"] = std_val
                
                print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                global_metrics[f"{key}_mean"] = None
                global_metrics[f"{key}_std"] = None
                print(f"{key}: No data available")
                
        return global_metrics


    def evaluate_loso(self):
        """
        Calculate LOSO metrics for each user and image, human ceiling,
        and prepare data for statistical tests.
        CORRECT: Apply LOSO to both humans and model.
        """
        loso_records = []
        users = [u for u in os.listdir(self.user_root) if u.startswith('participant_') and os.path.isdir(os.path.join(self.user_root, u))]

        for img_id in range(1, self.n_prompts + 1):
            for suffix in self.suffixes:
                # Load maps of all users for this image
                saliency_stack = {}
                for user in users:
                    maps = self._collect_saliency_maps_for_user(user, img_id, suffix)
                    if maps is not None:
                        saliency_stack[user] = maps
                
                # Need at least 2 users to do LOSO
                if len(saliency_stack) < 2:
                    print(f"Warning: Only {len(saliency_stack)} users for img_{img_id}{suffix}, skipping...")
                    continue

                # ==============================================
                # LOSO for HUMANS (already correct)
                # ==============================================
                for user in saliency_stack:
                    others = [arr for u, arr in saliency_stack.items() if u != user]
                    gt_loso = np.mean(np.stack(others), axis=0)
                    

                    rec = {
                        "img_id": img_id,
                        "suffix": suffix,
                        "source": "human",
                        "user": user,
                        "left_out_user": user,  # Added for clarity
                        "CC": self.cc(saliency_stack[user], gt_loso),
                        "KL": self.kl(saliency_stack[user], gt_loso),
                        "SIM": self.sim(saliency_stack[user], gt_loso)
                    }
                    loso_records.append(rec)

                # ==============================================
                # LOSO for MODEL (NEW - CORRECT)
                # ==============================================
                # For 'full' suffix, use '5000' for synthetics
                synthetic_suffix = '5000' if suffix == 'full' else suffix
                synthetic = self._load_synthetic_map(img_id, suffix, 'salmaps_arrays')
                if synthetic is not None:
                    # For each user we "leave out", compare the model
                    # with the average of OTHER users (same approach as humans)
                    for left_out_user in saliency_stack:
                        others = [arr for u, arr in saliency_stack.items() if u != left_out_user]
                        gt_loso = np.mean(np.stack(others), axis=0)
                        
                        # Resize the model to GT dimensions
                        synthetic_resized = cv2.resize(synthetic, (gt_loso.shape[1], gt_loso.shape[0]))
                    
                        
                        rec = {
                            "img_id": img_id,
                            "suffix": suffix,
                            "source": "model",
                            "user": f"model",  # Always "model"
                            "left_out_user": left_out_user,  # Which user was excluded from GT
                            "CC": self.cc(synthetic_resized, gt_loso),
                            "KL": self.kl(synthetic_resized, gt_loso),
                            "SIM": self.sim(synthetic_resized, gt_loso)
                        }
                        loso_records.append(rec)
                else:
                    print(f"Warning: Synthetic map not found for img_{img_id}{suffix} (using synthetic_{synthetic_suffix})")

        # Save all LOSO results
        df_loso = pd.DataFrame(loso_records)
        df_loso.to_csv(os.path.join(self.metrics_dir, 'metrics_loso.csv'), index=False)
        print(f"LOSO results saved: {len(df_loso)} records")

        # Calculate human ceiling (average of human LOSO performance)
        human_ceiling = df_loso[df_loso['source'] == 'human'].groupby('suffix')[['CC','KL','SIM']].mean().reset_index()
        human_ceiling.to_csv(os.path.join(self.metrics_dir, 'human_ceiling.csv'), index=False)
        print("Human ceiling saved.")

        # Run correct statistical tests
        self._run_stat_tests(df_loso)
    
        return df_loso  # Return for any additional analysis

    def _collect_saliency_maps_for_user(self, user, img_id, suffix):
        """
        Returns the NPY saliency map for a specific user and image.
        """
        user_path0 = os.path.join(self.user_root, user)
        user_path = os.path.join(user_path0, 'session_1')
        for sub_dir in os.listdir(user_path):
            if 'results_' in sub_dir and os.path.isdir(os.path.join(user_path, sub_dir)):
                saliency_path = os.path.join(
                    user_path, sub_dir, "salmaps_array",
                    f'img_prompt_{img_id}_saliency_array_{suffix}.npy'
                )
                if os.path.isfile(saliency_path):
                    return np.load(saliency_path).astype(np.float32)
        return None

    def _run_stat_tests(self, df_loso):
        """
        Run statistical tests between model and humans for each metric.
        CORRECT: Compare average LOSO performance per image+suffix.
        """
        results = []
        
        for metric in ['CC','KL','SIM']:
            print(f"\n=== Statistical tests for {metric} ===")
            
            # CORRECT AGGREGATION:
            # For each combination (img_id, suffix), calculate the average of LOSO performance
            
            # Human performance: average of all humans for each (img_id, suffix)
            df_hum_agg = df_loso[df_loso['source'] == 'human'].groupby(['img_id','suffix'])[metric].mean().reset_index()
            df_hum_agg.rename(columns={metric: f"{metric}_human"}, inplace=True)
            
            # Model performance: average of all model LOSO evaluations for each (img_id, suffix)
            df_mod_agg = df_loso[df_loso['source'] == 'model'].groupby(['img_id','suffix'])[metric].mean().reset_index()
            df_mod_agg.rename(columns={metric: f"{metric}_model"}, inplace=True)
            
            # Merge to have aligned pairs
            df_pair = pd.merge(df_hum_agg, df_mod_agg, on=['img_id','suffix'], how='inner')
            
            if len(df_pair) == 0:
                print(f"Warning: No valid pairs for {metric}")
                continue
            
            hum_vals = df_pair[f"{metric}_human"].values
            mod_vals = df_pair[f"{metric}_model"].values
            
            print(f"N pairs: {len(df_pair)}")
            print(f"Human mean: {hum_vals.mean():.4f} ± {hum_vals.std():.4f}")
            print(f"Model mean: {mod_vals.mean():.4f} ± {mod_vals.std():.4f}")
            
            # Differences
            diffs = hum_vals - mod_vals
            print(f"Mean difference (humans - model): {diffs.mean():.4f}")
            
            # Normality test of differences
            if len(diffs) >= 3:
                stat_shapiro, p_shapiro = shapiro(diffs)
                print(f"Normality test (Shapiro): p = {p_shapiro:.4f}")
            else:
                stat_shapiro, p_shapiro = None, None
                print("Too few samples for normality test")

            # Choose appropriate test
            if p_shapiro is not None and p_shapiro > 0.05 and len(diffs) >= 3:
                test_type = "paired_t"
                stat, p_val = ttest_rel(hum_vals, mod_vals)
                print(f"Paired t-test: t = {stat:.4f}, p = {p_val:.4f}")
            else:
                test_type = "wilcoxon"
                if len(diffs) >= 6:  # Wilcoxon needs at least 6 pairs
                    stat, p_val = wilcoxon(hum_vals, mod_vals)
                    print(f"Wilcoxon test: W = {stat:.4f}, p = {p_val:.4f}")
                else:
                    stat, p_val = None, None
                    print("Too few samples for Wilcoxon test")

            # Effect size (Cohen's d per paired data)
            if len(diffs) > 1:
                cohens_d = diffs.mean() / diffs.std()
            else:
                cohens_d = None

            results.append({
                "metric": metric,
                "test_type": test_type,
                "stat": stat,
                "p_value": p_val,
                "shapiro_p": p_shapiro,
                "mean_human": hum_vals.mean(),
                "mean_model": mod_vals.mean(),
                "mean_diff": diffs.mean(),
                "std_diff": diffs.std(),
                "cohens_d": cohens_d,
                "n_pairs": len(diffs),
                "n_human_obs": len(df_loso[df_loso['source'] == 'human']),
                "n_model_obs": len(df_loso[df_loso['source'] == 'model'])
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(self.metrics_dir, 'stat_tests_results.csv'), index=False)
        print(f"\nStatistical tests saved to: {os.path.join(self.metrics_dir, 'stat_tests_results.csv')}")
        
        # Print summary
        print("\n=== RESULTS SUMMARY ===")
        for _, row in df_results.iterrows():
            significance = "***" if row['p_value'] and row['p_value'] < 0.001 else \
                        "**" if row['p_value'] and row['p_value'] < 0.01 else \
                        "*" if row['p_value'] and row['p_value'] < 0.05 else "ns"
            
            print(f"{row['metric']}: Humans={row['mean_human']:.3f}, Model={row['mean_model']:.3f}, "
                f"Diff={row['mean_diff']:.3f}, p={row['p_value']:.3f} {significance}")

        return df_results

    def _run_mixed_effects(self, df_loso):
        """
        Runs a linear mixed effects model: metric ~ source + (1|img_id) + (1|left_out_user)
        NOTE: Uses left_out_user as random effect instead of user for the model.
        """
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            print("\n=== MIXED EFFECTS MODELS ===")
        except ImportError:
            print("Warning: statsmodels not available, skipping mixed effects")
            return

        for metric in ['CC','KL','SIM']:
            print(f"\n--- Mixed Effects for {metric} ---")
            
            try:
                # Prepare data for mixed effects
                model_data = df_loso.copy()
                
                # Create categorical variables
                model_data['source_cat'] = pd.Categorical(model_data['source'], categories=['human', 'model'])
                model_data['img_id_cat'] = pd.Categorical(model_data['img_id'])
                model_data['left_out_user_cat'] = pd.Categorical(model_data['left_out_user'])
                
                # Check that we have enough data
                print(f"Total observations: {len(model_data)}")
                print(f"Humans: {len(model_data[model_data['source'] == 'human'])}")
                print(f"Model: {len(model_data[model_data['source'] == 'model'])}")
                print(f"Unique images: {model_data['img_id'].nunique()}")
                print(f"Unique left-out users: {model_data['left_out_user'].nunique()}")
                
                if len(model_data) < 10:
                    print(f"Warning: Too few data points ({len(model_data)}) for mixed effects")
                    continue
                
                # Model with random intercepts for img_id and left_out_user
                formula = f"{metric} ~ source_cat"
                md = smf.mixedlm(formula, model_data, groups=model_data["img_id_cat"])
                
                # Add random effect for left_out_user if there are enough levels
                if model_data['left_out_user'].nunique() > 2:
                    md = smf.mixedlm(formula, model_data, 
                                groups=model_data["img_id_cat"], 
                                re_formula="~1",
                                vc_formula={"left_out_user_cat": "0 + left_out_user_cat"})
                
                mdf = md.fit(method='nm')  # Nelder-Mead is often more stable
                
                print(f"\nModel: {formula}")
                print("=" * 50)
                print(mdf.summary())
                
                # Extract main results
                fixed_effects = mdf.fe_params
                pvalues = mdf.pvalues
                
                print(f"\nKEY RESULTS for {metric}:")
                print(f"Intercept (human baseline): {fixed_effects.iloc[0]:.4f}")
                if len(fixed_effects) > 1:
                    model_effect = fixed_effects.iloc[1]
                    model_pval = pvalues.iloc[1]
                    direction = "better" if model_effect > 0 else "worse"
                    significance = "***" if model_pval < 0.001 else "**" if model_pval < 0.01 else "*" if model_pval < 0.05 else "ns"
                    
                    print(f"Model effect: {model_effect:.4f} (p = {model_pval:.4f}) {significance}")
                    print(f"→ The model is {direction} than humans by {abs(model_effect):.4f} points")
                    
            except Exception as e:
                print(f"Error in mixed effects for {metric}: {str(e)}")
                print("Possible causes: failed convergence or too little data")
                continue
                
        print("\n" + "="*50)
    
    def analyze_loso_results(self, df_loso):
        """
        Analyzes LOSO results and produces useful descriptive statistics.
        """
        print("\n" + "="*60)
        print("LOSO RESULTS ANALYSIS")
        print("="*60)
        
        # 1. General statistics
        print("\n1. GENERAL STATISTICS:")
        print(f"Total observations: {len(df_loso)}")
        print(f"Human observations: {len(df_loso[df_loso['source'] == 'human'])}")
        print(f"Model observations: {len(df_loso[df_loso['source'] == 'model'])}")
        print(f"Images: {df_loso['img_id'].nunique()}")
        print(f"Suffixes: {df_loso['suffix'].nunique()}")
        print(f"Unique users: {df_loso['left_out_user'].nunique()}")
        
        # 2. Mean performance per source
        print("\n2. MEAN PERFORMANCE:")
        summary_stats = df_loso.groupby('source')[['CC','KL','SIM']].agg(['mean', 'std', 'count'])
        print(summary_stats.round(4))
        
        # 3. Performance by suffix
        print("\n3. PERFORMANCE BY SUFFIX:")
        suffix_stats = df_loso.groupby(['source', 'suffix'])[['CC','KL','SIM']].mean()
        print(suffix_stats.round(4))
        print("\n3. PERFORMANCE BY SUFFIX and img_id:")
        img_means = df_loso.groupby(['source','suffix','img_id'])[['CC','KL','SIM']].mean()

        # then mean of images
        suffix_stats = img_means.groupby(['source','suffix'])[['CC','KL','SIM']].mean()
        print(suffix_stats.round(4))
        
        # 4. Variability across images
        print("\n4. VARIABILITY ACROSS IMAGES (top/bottom 3):")
        for metric in ['CC', 'KL', 'SIM']:
            img_means = df_loso.groupby(['source', 'img_id'])[metric].mean().unstack(level=0)
            if 'human' in img_means.columns and 'model' in img_means.columns:
                # Images where model does better
                diff = img_means['model'] - img_means['human'] 
                print(f"\n{metric} - Images where MODEL does better:")
                best_imgs = diff.nlargest(3)
                for img_id, diff_val in best_imgs.items():
                    h_val = img_means.loc[img_id, 'human']
                    m_val = img_means.loc[img_id, 'model'] 
                    print(f"  Img {img_id}: H={h_val:.3f}, M={m_val:.3f}, Diff=+{diff_val:.3f}")
                    
                print(f"{metric} - Images where HUMANS do better:")
                worst_imgs = diff.nsmallest(3)
                for img_id, diff_val in worst_imgs.items():
                    h_val = img_means.loc[img_id, 'human']
                    m_val = img_means.loc[img_id, 'model']
                    print(f"  Img {img_id}: H={h_val:.3f}, M={m_val:.3f}, Diff={diff_val:.3f}")
        
        # 5. Correlations between metrics
        print("\n5. CORRELATIONS BETWEEN METRICS:")
        for source in ['human', 'model']:
            corr_matrix = df_loso[df_loso['source'] == source][['CC','KL','SIM']].corr()
            print(f"\n{source.upper()}:")
            print(corr_matrix.round(3))
        
        # 6. Model consistency
        print("\n6. MODEL CONSISTENCY:")
        if 'model' in df_loso['source'].values:
            model_consistency = df_loso[df_loso['source'] == 'model'].groupby('img_id')[['CC','KL','SIM']].std()
            human_consistency = df_loso[df_loso['source'] == 'human'].groupby('img_id')[['CC','KL','SIM']].std()
            
            print("Standard deviation per image (lower = more consistent):")
            print(f"Model: CC={model_consistency['CC'].mean():.4f}, KL={model_consistency['KL'].mean():.4f}, SIM={model_consistency['SIM'].mean():.4f}")
            print(f"Humans:   CC={human_consistency['CC'].mean():.4f}, KL={human_consistency['KL'].mean():.4f}, SIM={human_consistency['SIM'].mean():.4f}")
        
        # 7. Save detailed summary
        summary_detailed = []
        
        for source in df_loso['source'].unique():
            for suffix in df_loso['suffix'].unique():
                subset = df_loso[(df_loso['source'] == source) & (df_loso['suffix'] == suffix)]
                if len(subset) > 0:
                    for metric in ['CC', 'KL', 'SIM']:
                        summary_detailed.append({
                            'source': source,
                            'suffix': suffix, 
                            'metric': metric,
                            'mean': subset[metric].mean(),
                            'std': subset[metric].std(),
                            'min': subset[metric].min(),
                            'max': subset[metric].max(),
                            'count': len(subset)
                        })
        
        df_summary = pd.DataFrame(summary_detailed)
        summary_path = os.path.join(self.metrics_dir, 'loso_detailed_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"\nDetailed summary saved to: {summary_path}")
        
        return df_summary

    def analyze_human_by_suffix(self, df_loso=None):
        """
        Analyzes LOSO results for humans, computing the mean metric for each image and suffix.
        Only for human data.
        
        Args:
            df_loso: DataFrame with LOSO results (if None, loads from file)
            
        Returns:
            DataFrame: Aggregated results for humans for each image and suffix
        """
        if df_loso is None:
            # Load LOSO data if not provided
            loso_path = os.path.join(self.metrics_dir, 'metrics_loso.csv')
            if os.path.exists(loso_path):
                df_loso = pd.read_csv(loso_path)
            else:
                print("Error: metrics_loso.csv not found. Run evaluate_loso() first.")
                return None
        
        # Filter only human data
        human_data = df_loso[df_loso['source'] == 'human']
        
        if len(human_data) == 0:
            print("Error: No human data found in LOSO results.")
            return None
        
        print("\n" + "="*60)
        print("HUMAN ANALYSIS BY SUFFIX")
        print("="*60)
        
        # Compute mean for each combination (img_id, suffix, metric)
        human_img_means = []
        
        for img_id in human_data['img_id'].unique():
            for suffix in human_data['suffix'].unique():
                for metric in ['CC', 'KL', 'SIM']:
                    subset = human_data[(human_data['img_id'] == img_id) & 
                                      (human_data['suffix'] == suffix)]
                    if len(subset) > 0:
                        mean_val = subset[metric].mean()
                        std_val = subset[metric].std()
                        human_img_means.append({
                            'img_id': img_id,
                            'suffix': suffix,
                            'metric': metric,
                            'mean': mean_val,
                            'std': std_val,
                            'count': len(subset)
                        })
        
        df_human_means = pd.DataFrame(human_img_means)
        
        # Show results for each suffix
        print("\nRESULTS BY SUFFIX:")
        for suffix in df_human_means['suffix'].unique():
            print(f"\n--- Suffix {suffix} ---")
            suffix_data = df_human_means[df_human_means['suffix'] == suffix]
            
            for metric in ['CC', 'KL', 'SIM']:
                metric_data = suffix_data[suffix_data['metric'] == metric]
                if len(metric_data) > 0:
                    overall_mean = metric_data['mean'].mean()
                    overall_std = metric_data['mean'].std()
                    print(f"{metric}: Mean={overall_mean:.4f} ± {overall_std:.4f} (over {len(metric_data)} images)")
        
        # Save results
        human_means_path = os.path.join(self.metrics_dir, 'human_means_by_suffix.csv')
        df_human_means.to_csv(human_means_path, index=False)
        print(f"\nHuman results by suffix saved to: {human_means_path}")
        
        # Also create a summary by suffix
        suffix_summary = []
        for suffix in df_human_means['suffix'].unique():
            for metric in ['CC', 'KL', 'SIM']:
                metric_data = df_human_means[(df_human_means['suffix'] == suffix) & 
                                           (df_human_means['metric'] == metric)]
                if len(metric_data) > 0:
                    suffix_summary.append({
                        'suffix': suffix,
                        'metric': metric,
                        'mean': metric_data['mean'].mean(),
                        'std': metric_data['mean'].std(),
                        'min': metric_data['mean'].min(),
                        'max': metric_data['mean'].max(),
                        'n_images': len(metric_data)
                    })
        
        df_suffix_summary = pd.DataFrame(suffix_summary)
        suffix_summary_path = os.path.join(self.metrics_dir, 'human_suffix_summary.csv')
        df_suffix_summary.to_csv(suffix_summary_path, index=False)
        print(f"Summary by suffix saved to: {suffix_summary_path}")
        
        return df_human_means, df_suffix_summary

    def wilcoxon_loso_vs_global(self, df_loso=None):
        """
        Runs Wilcoxon tests to compare LOSO metrics vs global metrics.
        Aggregates data by image and then does a SINGLE test for each metric and suffix.
        
        Args:
            df_loso: DataFrame with LOSO results (if None, loads from file)
            
        Returns:
            DataFrame: Wilcoxon test results
        """
        if df_loso is None:
            # Load LOSO data if not provided
            loso_path = os.path.join(self.metrics_dir, 'metrics_loso.csv')
            if os.path.exists(loso_path):
                df_loso = pd.read_csv(loso_path)
            else:
                print("Error: metrics_loso.csv not found. Run evaluate_loso() first.")
                return None
        
        # Load global data
        global_path = os.path.join(self.metrics_dir, 'metrics_results.csv')
        if os.path.exists(global_path):
            df_global = pd.read_csv(global_path)
        else:
            print("Error: metrics_results.csv not found.")
            return None
        
        print("\n" + "="*60)
        print("WILCOXON TEST: LOSO vs GLOBAL (AGGREGATED)")
        print("="*60)
        
        wilcoxon_results = []
        
        # For each combination (suffix, metric)
        for suffix in df_loso['suffix'].unique():
            for metric in ['CC', 'KL', 'SIM']:
                
                # Aggregate LOSO data by image (mean for each image)
                loso_agg = df_loso[(df_loso['suffix'] == suffix) & 
                                 (df_loso['source'] == 'human')].groupby('img_id')[metric].mean()
                
                # Aggregate global data by image (already aggregated, take the value)
                # For 'full' suffix, compare with '5000' of synthetics
                global_suffix = '5000' if suffix == 'full' else suffix
                global_agg = df_global[df_global['suffix'] == global_suffix].set_index('img_id')[metric]
                
                # Find common images
                common_images = loso_agg.index.intersection(global_agg.index)
                
                if len(common_images) >= 6:  # At least 6 pairs for Wilcoxon
                    # Take values for common images
                    loso_values = loso_agg.loc[common_images].values
                    global_values = global_agg.loc[common_images].values
                    
                    try:
                        # Wilcoxon test
                        from scipy.stats import wilcoxon
                        stat, p_value = wilcoxon(loso_values, global_values)
                        
                        # Compute descriptive statistics
                        loso_mean = loso_values.mean()
                        loso_std = loso_values.std()
                        global_mean = global_values.mean()
                        global_std = global_values.std()
                        
                        # Compute effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(loso_values) - 1) * loso_std**2 + 
                                            (len(global_values) - 1) * global_std**2) / 
                                           (len(loso_values) + len(global_values) - 2))
                        cohens_d = (loso_mean - global_mean) / pooled_std if pooled_std > 0 else 0
                        
                        result = {
                            'suffix': suffix,
                            'metric': metric,
                            'loso_mean': loso_mean,
                            'loso_std': loso_std,
                            'global_mean': global_mean,
                            'global_std': global_std,
                            'mean_diff': loso_mean - global_mean,
                            'wilcoxon_stat': stat,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'n_images': len(common_images)
                        }
                        
                        wilcoxon_results.append(result)
                        
                    except Exception as e:
                        print(f"Error in test for suffix {suffix}, {metric}: {str(e)}")
                else:
                    print(f"Too few data for suffix {suffix}, {metric}: {len(common_images)} images")
        
        # Create DataFrame with results
        df_wilcoxon = pd.DataFrame(wilcoxon_results)
        
        if len(df_wilcoxon) > 0:
            # Apply FDR (False Discovery Rate) correction
            from statsmodels.stats.multitest import multipletests
            
            # Extract p-values
            p_values = df_wilcoxon['p_value'].values
            
            # Apply FDR correction (Benjamini-Hochberg)
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, 
                alpha=0.05, 
                method='fdr_bh'
            )
            
            # Add corrected results to DataFrame
            df_wilcoxon['p_corrected'] = p_corrected
            df_wilcoxon['rejected'] = rejected
            
            # Determine significance for original and corrected p-values
            df_wilcoxon['significance_raw'] = df_wilcoxon['p_value'].apply(
                lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else "ns"
            )
            df_wilcoxon['significance_fdr'] = df_wilcoxon['p_corrected'].apply(
                lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else "ns"
            )
            
            # Save results
            wilcoxon_path = os.path.join(self.metrics_dir, 'wilcoxon_loso_vs_global.csv')
            df_wilcoxon.to_csv(wilcoxon_path, index=False)
            print(f"\nWilcoxon results saved to: {wilcoxon_path}")
            
            # Summary
            print("\nSUMMARY:")
            significant_raw = len(df_wilcoxon[df_wilcoxon['p_value'] < 0.05])
            significant_fdr = len(df_wilcoxon[df_wilcoxon['p_corrected'] < 0.05])
            total_count = len(df_wilcoxon)
            print(f"Significant tests (raw): {significant_raw}/{total_count} ({significant_raw/total_count*100:.1f}%)")
            print(f"Significant tests (FDR): {significant_fdr}/{total_count} ({significant_fdr/total_count*100:.1f}%)")
            
            # Show all results
            print("\nALL RESULTS:")
            for _, row in df_wilcoxon.iterrows():
                print(f"{row['metric']} (suffix {row['suffix']}): "
                      f"diff={row['mean_diff']:.4f}, "
                      f"p_raw={row['p_value']:.4f} {row['significance_raw']}, "
                      f"p_fdr={row['p_corrected']:.4f} {row['significance_fdr']}")
            
            # Show only significant results after FDR correction
            print("\nSIGNIFICANT RESULTS AFTER FDR CORRECTION:")
            significant_results = df_wilcoxon[df_wilcoxon['p_corrected'] < 0.05]
            if len(significant_results) > 0:
                for _, row in significant_results.iterrows():
                    print(f"{row['metric']} (suffix {row['suffix']}): "
                          f"LOSO={row['loso_mean']:.4f}±{row['loso_std']:.4f}, "
                          f"Global={row['global_mean']:.4f}±{row['global_std']:.4f}, "
                          f"p_fdr={row['p_corrected']:.4f} {row['significance_fdr']}")
            else:
                print("No significant results after FDR correction.")
            
            return df_wilcoxon
        else:
            print("No valid results found.")
            return None

    def create_metric_boxplots(self, save_plots=True):
        """
        Creates boxplots for each metric (CC, KL, SIM) comparing model vs human average.
        Layout identical to rm_analyzer.py with seaborn boxplot.
        
        Args:
            save_plots: If True, saves the plots as PNG files
            
        Returns:
            dict: Dictionary with the created plots
        """
        # Load global data (model vs human average)
        global_path = os.path.join(self.metrics_dir, 'metrics_results.csv')
        if os.path.exists(global_path):
            df_global = pd.read_csv(global_path)
        else:
            print("Error: metrics_results.csv not found. Run global evaluation first.")
            return None
        
        print("\n" + "="*60)
        print("CREATING BOXPLOTS: MODEL vs HUMAN AVERAGE")
        print("="*60)
        
        # Import matplotlib
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Error: matplotlib and seaborn are required for plotting.")
            return None
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(self.metrics_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Define the metrics to plot
        metrics = {
            'CC': 'CC',
            'KL': 'KL', 
            'SIM': 'SIM'
        }
        
        # Create figure with subplots arranged horizontally
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle('Model vs Human Average', fontsize=18, fontweight='bold')
        
        for idx, (metric_name, column_name) in enumerate(metrics.items()):
            ax = axes[idx]
            
            # Prepare data for boxplot
            # For each suffix, compare model vs human average
            boxplot_data = []
            labels = []
            
            # Order suffixes in the specific order: 500, 3000, 5000, full
            suffix_order = ['500', '3000', '5000', 'full']
            for suffix in suffix_order:
                suffix_data = df_global[df_global['suffix'] == suffix]
                
                if len(suffix_data) > 0:
                    # Data is already aggregated by image (model vs human average)
                    values = suffix_data[metric_name].values
                    boxplot_data.append(values)
                    labels.append(suffix)
                    print(f"  Time window {suffix}ms: {len(values)} images, mean={values.mean():.4f}±{values.std():.4f}")
            
            if boxplot_data:
                # Prepare long-form data for seaborn
                df_plot = pd.DataFrame()
                for i, (suffix, data) in enumerate(zip(labels, boxplot_data)):
                    suffix_df = pd.DataFrame({
                        'Value': data,
                        'Time window (ms)': [suffix] * len(data),
                        'Image': [f'img_{j+1}' for j in range(len(data))]
                    })
                    df_plot = pd.concat([df_plot, suffix_df], ignore_index=True)
                
                # Create boxplot on the current subplot
                sns.boxplot(data=df_plot, x='Time window (ms)', y='Value', ax=ax, palette='Blues', width=0.5)
                
                # Add individual points
                for i, suffix in enumerate(labels):
                    suffix_data = df_plot[df_plot['Time window (ms)'] == suffix]
                    x_pos = i
                    ax.plot([x_pos] * len(suffix_data), suffix_data['Value'], 'o', 
                           color='darkblue', alpha=0.6, markersize=3)
                
                # Configure titles and labels with larger font sizes
                ax.set_title(f"{metric_name}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Time window (ms)", fontsize=12)
                ax.set_ylabel(f"{metric_name}", fontsize=12)
                
                # Increase tick label font sizes
                ax.tick_params(axis='both', which='major', labelsize=11)
                # Add padding to move x-axis labels further from the axes
                ax.tick_params(axis='x', pad=15)
        
        plt.tight_layout()
        
        # Save the plot
        if save_plots:
            plot_path = os.path.join(plots_dir, 'model_vs_human_boxplot.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  Plot saved to: {plot_path}")
        
        plt.show()
        
        print(f"\nPlot created and saved in: {plots_dir}")
        return {'combined': fig}