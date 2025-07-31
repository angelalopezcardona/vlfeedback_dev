import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class MetricsAnalyzer:
    def __init__(self, users_dir="users"):
        self.users_dir = users_dir
        self.aggregate_data = None
        self.participants = []
        
    def get_participant_directories(self):
        """Get all participant directories"""
        participant_dirs = []
        for item in os.listdir(self.users_dir):
            item_path = os.path.join(self.users_dir, item)
            if os.path.isdir(item_path) and item.startswith('participant_'):
                participant_dirs.append(item)
        return sorted(participant_dirs)
    
    def read_participant_data(self, participant_dir):
        """Read general_features.csv and info_extended.csv for a participant"""
        session_path = os.path.join(self.users_dir, participant_dir, "session_1")
        
        # Read general features
        general_features_path = os.path.join(session_path, "general_features.csv")
        if not os.path.exists(general_features_path):
            print(f"Warning: {general_features_path} not found")
            return None
            
        general_features = pd.read_csv(general_features_path, sep=';')
        
        # Read info extended to get the 'best' and 'discarted' columns
        info_extended_path = os.path.join(session_path, "info_extended.csv")
        if not os.path.exists(info_extended_path):
            print(f"Warning: {info_extended_path} not found")
            return None
            
        info_extended = pd.read_csv(info_extended_path)
        
        # Extract participant name
        participant_name = participant_dir.replace('participant_', '').replace('_', ' ')
        
        # Merge the dataframes on trial column
        merged_data = pd.merge(general_features, info_extended[['trial', 'best', 'discarted']], 
                              on='trial', how='inner')
        
        # Add participant information
        merged_data['participant'] = participant_name
        merged_data['participant_id'] = participant_dir
        
        return merged_data
    
    def aggregate_all_data(self):
        """Aggregate data from all participants"""
        participant_dirs = self.get_participant_directories()
        all_data = []
        
        for participant_dir in participant_dirs:
            print(f"Processing {participant_dir}...")
            data = self.read_participant_data(participant_dir)
            if data is not None:
                all_data.append(data)
                self.participants.append(participant_dir.replace('participant_', '').replace('_', ' '))
        
        if all_data:
            self.aggregate_data = pd.concat(all_data, ignore_index=True)
            
            # Filter out trials with .0 values (keep only .1 and .2 trials)
            print(f"Before filtering: {len(self.aggregate_data)} trials")
            
            # Convert trial to string to check for .0, .1, .2 endings
            self.aggregate_data['trial_str'] = self.aggregate_data['trial'].astype(str)
            
            # Keep only trials ending with .1 or .2
            self.aggregate_data = self.aggregate_data[
                self.aggregate_data['trial_str'].str.endswith('.1') | 
                self.aggregate_data['trial_str'].str.endswith('.2')
            ]
            
            # Remove discarded trials
            self.aggregate_data = self.aggregate_data[self.aggregate_data['discarted'] == False]
            
            # Remove the temporary trial_str column
            self.aggregate_data = self.aggregate_data.drop('trial_str', axis=1)
            
            # Convert time-based metrics from milliseconds to seconds
            time_metrics = ['first_fix_duration_mean', 'fix_duration_mean', 'go_past_time_mean']
            for metric in time_metrics:
                if metric in self.aggregate_data.columns:
                    self.aggregate_data[metric] = self.aggregate_data[metric] / 1000.0
                    print(f"Converted {metric} from milliseconds to seconds")
            
            print(f"After filtering (.1/.2 trials, non-discarded): {len(self.aggregate_data)} trials")
            print(f"Aggregated data shape: {self.aggregate_data.shape}")
            print(f"Participants processed: {len(self.participants)}")
            return self.aggregate_data
        else:
            print("No data found!")
            return None
    
    def save_aggregate_data(self, filename="aggregate_metrics.csv"):
        """Save the aggregated data to CSV"""
        if self.aggregate_data is not None:
            self.aggregate_data.to_csv(os.path.join(self.users_dir, filename), index=False)
            print(f"Aggregate data saved to {filename}")
        else:
            print("No data to save!")
    
    def perform_paired_t_tests(self):
        """Perform paired t-tests and Cohen's d for TRT, FFD, NFIX"""
        if self.aggregate_data is None:
            print("No data available for analysis!")
            return None
        
        # Define the metrics to analyze
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }
        
        results = {}
        
        for metric_name, column_name in metrics.items():
            print(f"\n=== Analysis for {metric_name} ({column_name}) ===")
            
            # STEP 1 – Prepare the 'trial_base' column (e.g., "30" from "30.1")
            df = self.aggregate_data.copy()
            df['trial_base'] = df['trial'].astype(str).str.split('.').str[0]
            
            # STEP 2 – Pivot to create "Preferred" and "Rejected" columns
            paired_df = df.pivot_table(
                index=['participant_id', 'trial_base'],
                columns='best',  # True = preferred, False = rejected
                values=column_name
            ).dropna()
            
            # Rename columns
            paired_df.columns = ['Rejected', 'Preferred']
            
            print(f"Paired observations (n={len(paired_df)}):")
            print(f"Preferred responses: mean={paired_df['Preferred'].mean():.2f}, std={paired_df['Preferred'].std():.2f}")
            print(f"Rejected responses: mean={paired_df['Rejected'].mean():.2f}, std={paired_df['Rejected'].std():.2f}")
            
            if len(paired_df) > 0:
                # STEP 3 – Perform paired t-test
                t_stat, p_value = stats.ttest_rel(paired_df['Preferred'], paired_df['Rejected'])
                
                # STEP 4 – Calculate Cohen's d for paired data
                diff = paired_df['Preferred'] - paired_df['Rejected']
                cohens_d = diff.mean() / diff.std(ddof=1)
                
                # Determine effect size interpretation
                if abs(cohens_d) < 0.2:
                    effect_size_interpretation = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_size_interpretation = "small"
                elif abs(cohens_d) < 0.8:
                    effect_size_interpretation = "medium"
                else:
                    effect_size_interpretation = "large"
                
                results[metric_name] = {
                    'preferred_mean': paired_df['Preferred'].mean(),
                    'preferred_std': paired_df['Preferred'].std(),
                    'rejected_mean': paired_df['Rejected'].mean(),
                    'rejected_std': paired_df['Rejected'].std(),
                    'paired_n': len(paired_df),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'effect_size_interpretation': effect_size_interpretation,
                    'significant': p_value < 0.05,
                    'mean_difference': diff.mean(),
                    'std_difference': diff.std(ddof=1)
                }
                
                print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
                print(f"Cohen's d: {cohens_d:.3f} ({effect_size_interpretation} effect)")
                print(f"Mean difference (Preferred - Rejected): {diff.mean():.3f}")
                print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
            else:
                print(f"Insufficient paired data for {metric_name}")
                results[metric_name] = None
        
        return results
    
    def perform_wilcoxon_test(self):
        """Perform Wilcoxon signed-rank test for each metric."""
        if self.aggregate_data is None:
            print("No data available for Wilcoxon test!")
            return None

        from scipy.stats import wilcoxon

        # Define the metrics
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }

        wilcoxon_results = {}

        for metric_name, column_name in metrics.items():
            print(f"\n=== Wilcoxon Test for {metric_name} ({column_name}) ===")

            df = self.aggregate_data.copy()
            df['trial_base'] = df['trial'].astype(str).str.split('.').str[0]

            # Pivot to get paired values
            paired_df = df.pivot_table(
                index=['participant_id', 'trial_base'],
                columns='best',
                values=column_name
            ).dropna()

            if paired_df.shape[0] == 0:
                print("No paired data available for this metric.")
                wilcoxon_results[metric_name] = None
                continue

            paired_df.columns = ['Rejected', 'Preferred']

            # Calculate differences
            diffs = paired_df['Preferred'] - paired_df['Rejected']

            # Perform Wilcoxon signed-rank test
            try:
                stat, p_value = wilcoxon(diffs)
            except ValueError as e:
                print(f"Wilcoxon test failed: {e}")
                wilcoxon_results[metric_name] = None
                continue

            wilcoxon_results[metric_name] = {
                'paired_n': len(diffs),
                'mean_diff': diffs.mean(),
                'std_diff': diffs.std(ddof=1),
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            print(f"Mean difference: {diffs.mean():.4f}")
            print(f"Wilcoxon signed-rank test: W={stat:.3f}, p={p_value:.4f}")
            print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")

        return wilcoxon_results
    
    def perform_sign_test(self):
        """Perform Sign Test (based on binomial test) to check if accepted responses consistently show better values."""
        if self.aggregate_data is None:
            print("No data available for Sign Test!")
            return None

        from scipy.stats import binomtest  # or binom_test for older versions

        # Define the metrics
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }

        sign_test_results = {}

        for metric_name, column_name in metrics.items():
            print(f"\n=== Sign Test for {metric_name} ({column_name}) ===")

            df = self.aggregate_data.copy()
            df['trial_base'] = df['trial'].astype(str).str.split('.').str[0]

            # Pivot to get matched preferred and rejected values
            paired_df = df.pivot_table(
                index=['participant_id', 'trial_base'],
                columns='best',
                values=column_name
            ).dropna()

            if paired_df.shape[0] == 0:
                print("No paired data available for this metric.")
                sign_test_results[metric_name] = None
                continue

            paired_df.columns = ['Rejected', 'Preferred']

            # Compare values: does Preferred < Rejected?
            comparisons = paired_df['Preferred'] < paired_df['Rejected']
            n_total = len(comparisons)
            n_success = comparisons.sum()
            prop = n_success / n_total

            # Perform binomial test (two-sided or one-sided: here we test "less is better")
            binom_result = binomtest(n_success, n_total, p=0.5, alternative='greater')

            # Save results
            sign_test_results[metric_name] = {
                'n_total': n_total,
                'n_success': int(n_success),
                'proportion': prop,
                'p_value': binom_result.pvalue,
                'significant': binom_result.pvalue < 0.05
            }

            # Print results
            print(f"Accepted responses had better scores in {n_success}/{n_total} trials ({prop:.1%})")
            print(f"Binomial test p-value: {binom_result.pvalue:.4f}")
            print(f"Significant: {'Yes' if binom_result.pvalue < 0.05 else 'No'}")

        return sign_test_results
    
    def compute_word_count_correlations(self):
        """Compute Pearson and Spearman correlations between number of words and reading metrics."""
        if self.aggregate_data is None:
            print("No data available for correlation analysis!")
            return None

        from scipy.stats import pearsonr, spearmanr

        word_col = 'number_words_mean'
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean',
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }

        results = {}

        print("\n=== Correlation between number_words_mean and reading metrics ===")

        for metric_name, column_name in metrics.items():
            if word_col in self.aggregate_data.columns and column_name in self.aggregate_data.columns:
                x = self.aggregate_data[word_col]
                y = self.aggregate_data[column_name]

                # Remove NaNs
                mask = x.notna() & y.notna()
                x_valid = x[mask]
                y_valid = y[mask]

                if len(x_valid) < 2:
                    print(f"{metric_name}: insufficient data")
                    results[metric_name] = None
                    continue

                # Pearson
                pearson_corr, pearson_p = pearsonr(x_valid, y_valid)

                # Spearman
                spearman_corr, spearman_p = spearmanr(x_valid, y_valid)

                print(f"{metric_name}:")
                print(f"  Pearson  r = {pearson_corr:.3f}, p = {pearson_p:.4f}")
                print(f"  Spearman rho = {spearman_corr:.3f}, p = {spearman_p:.4f}")

                results[metric_name] = {
                    'pearson_r': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_rho': spearman_corr,
                    'spearman_p': spearman_p,
                    'n': len(x_valid)
                }
            else:
                print(f"{metric_name}: column not found")

        return results
    
    def save_statistical_results(self, results, filename="statistical_results.csv"):
        """Save paired t-test results to CSV"""
        if results is None:
            print("No results to save!")
            return
        
        # Convert results to DataFrame
        results_data = []
        for metric, result in results.items():
            if result is not None:
                results_data.append({
                    'Metric': metric,
                    'Preferred_Mean': result['preferred_mean'],
                    'Preferred_Std': result['preferred_std'],
                    'Rejected_Mean': result['rejected_mean'],
                    'Rejected_Std': result['rejected_std'],
                    'Paired_N': result['paired_n'],
                    'Mean_Difference': result['mean_difference'],
                    'Std_Difference': result['std_difference'],
                    'T_Statistic': result['t_statistic'],
                    'P_Value': result['p_value'],
                    'Cohens_D': result['cohens_d'],
                    'Effect_Size_Interpretation': result['effect_size_interpretation'],
                    'Significant': result['significant']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        print(f"Paired t-test results saved to {filename}")
        
        return results_df
    
    def save_wilcoxon_results(self, results, filename="wilcoxon_results.csv"):
        """Save Wilcoxon signed-rank test results to CSV"""
        if results is None:
            print("No Wilcoxon results to save!")
            return
        
        # Convert results to DataFrame
        results_data = []
        for metric, result in results.items():
            if result is not None:
                results_data.append({
                    'Metric': metric,
                    'Paired_N': result['paired_n'],
                    'Mean_Difference': result['mean_diff'],
                    'Std_Difference': result['std_diff'],
                    'W_Statistic': result['statistic'],
                    'P_Value': result['p_value'],
                    'Significant': result['significant']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        print(f"Wilcoxon test results saved to {filename}")
        
        return results_df
    
    def save_sign_test_results(self, results, filename="sign_test_results.csv"):
        """Save Sign test results to CSV"""
        if results is None:
            print("No Sign test results to save!")
            return
        
        # Convert results to DataFrame
        results_data = []
        for metric, result in results.items():
            if result is not None:
                results_data.append({
                    'Metric': metric,
                    'N_Total': result['n_total'],
                    'N_Success': result['n_success'],
                    'Proportion': result['proportion'],
                    'P_Value': result['p_value'],
                    'Significant': result['significant']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        print(f"Sign test results saved to {filename}")
        
        return results_df
    
    def save_correlation_results(self, results, filename="correlation_results.csv"):
        """Save correlation results to CSV"""
        if results is None:
            print("No correlation results to save!")
            return
        
        # Convert results to DataFrame
        results_data = []
        for metric, result in results.items():
            if result is not None:
                results_data.append({
                    'Metric': metric,
                    'N': result['n'],
                    'Pearson_r': result['pearson_r'],
                    'Pearson_p': result['pearson_p'],
                    'Spearman_rho': result['spearman_rho'],
                    'Spearman_p': result['spearman_p']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        print(f"Correlation results saved to {filename}")
        
        return results_df
    
    def print_summary_statistics(self):
        """Print summary statistics for the aggregated data"""
        if self.aggregate_data is None:
            print("No data available!")
            return
        
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total trials: {len(self.aggregate_data)}")
        print(f"Participants: {len(self.participants)}")
        print(f"Preferred responses: {len(self.aggregate_data[self.aggregate_data['best'] == True])}")
        print(f"Rejected responses: {len(self.aggregate_data[self.aggregate_data['best'] == False])}")
        
        # Print metrics summary
        metrics = ['first_fix_duration_mean', 'fix_duration_mean', 'fix_number_mean', 'go_past_time_mean']
        metric_names = ['FFD', 'TRT', 'nFIX', 'GPT']
        
        print("\nMetrics Summary:")
        for metric, name in zip(metrics, metric_names):
            if metric in self.aggregate_data.columns:
                data = self.aggregate_data[metric].dropna()
                print(f"{name}: mean={data.mean():.2f}, std={data.std():.2f}, n={len(data)}")
    
    def save_summary_statistics(self, filename="summary_statistics.txt", results=None, wilcoxon_results=None, sign_test_results=None, correlation_results=None):
        """Save summary statistics to a text file"""
        if self.aggregate_data is None:
            print("No data available!")
            return
        
        with open(filename, 'w') as f:
            f.write("=== SUMMARY STATISTICS ===\n")
            f.write(f"Total trials: {len(self.aggregate_data)}\n")
            f.write(f"Participants: {len(self.participants)}\n")
            f.write(f"Preferred responses: {len(self.aggregate_data[self.aggregate_data['best'] == True])}\n")
            f.write(f"Rejected responses: {len(self.aggregate_data[self.aggregate_data['best'] == False])}\n")
            
            # Write metrics summary
            metrics = ['first_fix_duration_mean', 'fix_duration_mean', 'fix_number_mean', 'go_past_time_mean']
            metric_names = ['FFD', 'TRT', 'nFIX', 'GPT']
            
            f.write("\nMetrics Summary:\n")
            for metric, name in zip(metrics, metric_names):
                if metric in self.aggregate_data.columns:
                    data = self.aggregate_data[metric].dropna()
                    f.write(f"{name}: mean={data.mean():.2f}, std={data.std():.2f}, n={len(data)}\n")
            
            # Write paired t-test results if available
            if results is not None:
                f.write("\n=== PAIRED T-TEST RESULTS ===\n")
                f.write("Comparison: Preferred vs Rejected Responses (Paired Analysis)\n\n")
                
                for metric_name, result in results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Paired observations (n={result['paired_n']})\n")
                        f.write(f"Preferred: mean={result['preferred_mean']:.2f}, std={result['preferred_std']:.2f}\n")
                        f.write(f"Rejected: mean={result['rejected_mean']:.2f}, std={result['rejected_std']:.2f}\n")
                        f.write(f"Mean difference (Preferred - Rejected): {result['mean_difference']:.3f}\n")
                        f.write(f"Paired t-test: t={result['t_statistic']:.3f}, p={result['p_value']:.4f}\n")
                        f.write(f"Cohen's d: {result['cohens_d']:.3f} ({result['effect_size_interpretation']} effect)\n")
                        f.write(f"Significant: {'Yes' if result['significant'] else 'No'}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient paired data for analysis\n\n")
            
            # Write Wilcoxon test results if available
            if wilcoxon_results is not None:
                f.write("\n=== WILCOXON SIGNED-RANK TEST RESULTS ===\n")
                f.write("Non-parametric alternative to paired t-test\n\n")
                
                for metric_name, result in wilcoxon_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Paired observations (n={result['paired_n']})\n")
                        f.write(f"Mean difference: {result['mean_diff']:.3f}\n")
                        f.write(f"Wilcoxon W statistic: {result['statistic']:.3f}\n")
                        f.write(f"P-value: {result['p_value']:.4f}\n")
                        f.write(f"Significant: {'Yes' if result['significant'] else 'No'}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient paired data for analysis\n\n")
            
            # Write Sign test results if available
            if sign_test_results is not None:
                f.write("\n=== SIGN TEST RESULTS ===\n")
                f.write("Tests if preferred responses consistently show better values\n\n")
                
                for metric_name, result in sign_test_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Total trials: {result['n_total']}\n")
                        f.write(f"Preferred better in: {result['n_success']} trials\n")
                        f.write(f"Proportion: {result['proportion']:.1%}\n")
                        f.write(f"P-value: {result['p_value']:.4f}\n")
                        f.write(f"Significant: {'Yes' if result['significant'] else 'No'}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient paired data for analysis\n\n")
            
            # Write correlation results if available
            if correlation_results is not None:
                f.write("\n=== CORRELATION RESULTS ===\n")
                f.write("Correlation between number of words and reading metrics\n\n")
                
                for metric_name, result in correlation_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Sample size: {result['n']}\n")
                        f.write(f"Pearson correlation: r={result['pearson_r']:.3f}, p={result['pearson_p']:.4f}\n")
                        f.write(f"Spearman correlation: rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.4f}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient data for correlation analysis\n\n")
            
            # Write participant list
            f.write(f"\nParticipants processed ({len(self.participants)}):\n")
            for participant in self.participants:
                f.write(f"- {participant}\n")
        
        print(f"Summary statistics saved to {filename}")
    
    def create_boxplots(self, results, output_dir="analysis_plots"):
        """Create one boxplot per metric comparing preferred vs rejected responses"""
        if self.aggregate_data is None:
            print("No data available for plots!")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the metrics to plot
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("\nCreating boxplots for each metric...")
        
        for metric_name, column_name in metrics.items():
            if column_name not in self.aggregate_data.columns:
                print(f"Warning: {column_name} not found in data")
                continue
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Prepare data for boxplot
            preferred_data = self.aggregate_data[self.aggregate_data['best'] == True][column_name].dropna()
            rejected_data = self.aggregate_data[self.aggregate_data['best'] == False][column_name].dropna()
            
            # Create boxplot
            data_to_plot = [preferred_data, rejected_data]
            labels = ['Preferred', 'Rejected']
            
            bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add title and labels
            plt.title(f'{metric_name}: Preferred vs Rejected Responses', fontsize=14, fontweight='bold')
            plt.ylabel(f'{metric_name} ({self._get_metric_unit(metric_name)})', fontsize=12)
            plt.xlabel('Response Type', fontsize=12)
            
            # Add sample sizes to labels
            plt.xticks([1, 2], [f'Preferred\n(n={len(preferred_data)})', f'Rejected\n(n={len(rejected_data)})'])
            

            
            # Improve layout
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plot_filename = os.path.join(output_dir, f'{metric_name}_boxplot.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Boxplot saved: {plot_filename}")
        
        print(f"\nBoxplots created in {output_dir}:")
        print("- FFD_boxplot.png: First Fixation Duration")
        print("- TRT_boxplot.png: Total Reading Time")
        print("- nFIX_boxplot.png: Number of Fixations")
        print("- GPT_boxplot.png: Go-Past Time")
    
    def _get_metric_unit(self, metric_name):
        """Get the appropriate unit for each metric"""
        units = {
            'FFD': 'seconds',
            'TRT': 'seconds', 
            'nFIX': 'count',
            'GPT': 'seconds'
        }
        return units.get(metric_name, '')

def main():
    """Main function to run the analysis"""
    print("Starting Metrics Analysis...")
    
    # Initialize analyzer
    analyzer = MetricsAnalyzer(users_dir="./fixations")
    
    # Aggregate all data
    print("\n1. Aggregating data from all participants...")
    aggregate_data = analyzer.aggregate_all_data()
    
    if aggregate_data is not None:
        # Save aggregate data
        print("\n2. Saving aggregate data...")
        analyzer.save_aggregate_data("aggregate_metrics.csv")
        
        # Print summary statistics
        print("\n3. Summary statistics...")
        analyzer.print_summary_statistics()
        
        # Perform statistical analysis
        print("\n4. Performing statistical analysis...")
        results = analyzer.perform_paired_t_tests()
        
        # Perform Wilcoxon test
        print("\n5. Performing Wilcoxon signed-rank test...")
        wilcoxon_results = analyzer.perform_wilcoxon_test()
        
        # Perform Sign test
        print("\n6. Performing Sign test...")
        sign_test_results = analyzer.perform_sign_test()
        
        # Compute word count correlations
        print("\n7. Computing word count correlations...")
        correlation_results = analyzer.compute_word_count_correlations()
        
        # Save summary statistics with all statistical results
        print("\n8. Saving summary statistics...")
        analyzer.save_summary_statistics("summary_statistics.txt", results, wilcoxon_results, sign_test_results, correlation_results)
        
        # Save all statistical results to separate CSV files
        print("\n9. Saving statistical results...")
        analyzer.save_statistical_results(results, "paired_t_test_results.csv")
        analyzer.save_wilcoxon_results(wilcoxon_results, "wilcoxon_results.csv")
        analyzer.save_sign_test_results(sign_test_results, "sign_test_results.csv")
        analyzer.save_correlation_results(correlation_results, "correlation_results.csv")
        
        # Create boxplots for each metric
        print("\n10. Creating boxplots for each metric...")
        analyzer.create_boxplots(results, "analysis_plots")
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Files created:")
        print("- aggregate_metrics.csv: All trial data with metrics")
        print("- summary_statistics.txt: Summary statistics with all statistical results")
        print("- paired_t_test_results.csv: Paired t-test results")
        print("- wilcoxon_results.csv: Wilcoxon signed-rank test results")
        print("- sign_test_results.csv: Sign test results")
        print("- correlation_results.csv: Correlation analysis results")
        print("- analysis_plots/: Boxplots for each metric")
        
    else:
        print("Failed to aggregate data. Please check the data structure.")

if __name__ == "__main__":
    main() 