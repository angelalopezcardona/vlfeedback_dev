#!/usr/bin/env python3
"""
Main script to run saliency evaluation.

This script evaluates and compares saliency maps from different eye-tracking devices 
(Tobii and Gazepoint) with synthetic saliency maps for 30 prompts.
"""

import sys
import os

# Add the fixation_visualizer directory to the path
import os
import sys

# If needed to maintain the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fixation_visualizer'))

from saliency_comparison import SaliencyEvaluator


def main():
    """
    Main function to run the saliency evaluation.
    """
    print("=" * 60)
    print("SALIENCY EVALUATION")
    print("=" * 60)
    print("Evaluating saliency maps for 30 prompts...")
    print("Processing suffixes: 500, 3000, 5000, full")
    print("Device types: Tobii, Gazepoint")
    print("=" * 60)
    
    try:
        # 1️⃣ Initialize and run global evaluation
        evaluator = SaliencyEvaluator(
            user_root="users",
            synthetic_salmaps_root="synthetic_salmaps", 
            original_images_root="images",
            output_dir="salmaps_avg",
            n_prompts=30
        )
        
        print("=" * 60)
        print("✅ Saliency evaluation (GLOBAL) completed successfully!")
        print("=" * 60)
        print("📁 Output directories created:")
        print("   - salmaps/salmaps/")
        print("   - salmaps/salmaps_vis/")
        print("   - salmaps/salmaps_arrays/")
        print("   - salmaps/metrics/")
        print("=" * 60)

        # 2️⃣ Run LOSO + human ceiling + statistical tests
        print("🔍 Running LOSO evaluation, human ceiling, and statistical tests...")
        df_loso = evaluator.evaluate_loso()  # Now returns the DataFrame
        #summary = evaluator.analyze_loso_results(df_loso)  # Detailed analysis
        #sos = evaluator._run_mixed_effects(df_loso)
        evaluator.analyze_human_by_suffix()
        wilcoxon_test = evaluator.wilcoxon_loso_vs_global()
        
        # Create boxplots of the metrics
        print("\n" + "="*60)
        print("CREATING METRIC BOXPLOTS")
        print("="*60)
        boxplots = evaluator.create_metric_boxplots()

        print("✅ LOSO evaluation & statistical tests completed.")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error during saliency evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
