    ## Compute attention of each model:
    - In this folder we can compute the attention of different transformer models and compare it with the reading measures.

    - Computing attention per models and per layer:
        run `python analyse_attention/main_compute_attention.py` to compute attention patterns across all models and layers.

        Key features:
        - Computes attention scores for each trial and layer across all configured models
        - Supports both standard attention computation and reward model attention computation
        - For reward model attention (run with `--reward=True`), computes attention on combined prompt+response
        - Reward model attention currently supported for:
            * openbmb/UltraRM-13b
            * openbmb/Eurus-RM-7b  
            * nicolinho/QRM-Llama3.1-8B

        Configuration:
        - Models can be enabled/disabled in the models dictionary in main_compute_attention.py
        - Results are saved per layer as CSV files in:
            * Standard attention: attention/model_name/results/set_n/trial_XX/layer_X.csv
            * Reward attention: attention_reward/model_name/results/set_n/trial_XX/layer_X.csv

        Example usage:
        ```bash
        # Standard attention computation
        python analyse_attention/main_compute_attention.py

        # Reward model attention computation 
        python analyse_attention/main_compute_attention.py --reward=True
        ```

    - Comparing attention with reading measures:
        This script analyzes correlations between model attention patterns and human reading measures.

        Usage:
        ```bash
        # Basic usage
        python analyse_attention/main_compute_compare_trials.py

        # Filter for unanimous responses only
        python analyse_attention/main_compute_compare_trials.py --filter_completed=True

        # Use reward model attention patterns
        python analyse_attention/main_compute_compare_trials.py --folder_attention=attention_reward
        ```

        Key features:
        - Computes correlations between model attention and reading measures (fixation duration, count etc.)
        - Can analyze either all responses or only unanimous ones via --filter_completed flag
        - Supports both standard attention and reward model attention via --folder_attention flag
        - Saves results separately for chosen and rejected responses
        - Results saved to:
            * Standard attention: attention/modelname/results/[chosen|rejected]/
            * Reward attention: attention_reward/modelname/results/[chosen|rejected]/

        Configuration:
        - Models can be enabled/disabled in the models dictionary in the script
        - Available reading measure can be configured in the gaze_features list
        - For reward models, only UltraRM-13b, Eurus-RM-7b and QRM-Llama3.1-8B are supported


    - Plotting attention-gaze correlations by model layer:
        ```bash
        python analyse_attention/main_plot_attention_layers.py
        ```

        Key features:
        - Visualizes correlations between model attention and reading measures for each layer
        - Configurable options (directly on the file):
            * Model selection
            * Reading measures to analyze (e.g. fixation duration, count)
            * Attention source folder (standard 'attention' or reward 'attention_reward')
        - Generates per-layer correlation plots to analyze attention patterns at different model depths

    - Plotting chosen vs rejected attention correlations:
        ```bash 
        python analyse_attention/main_plot_chosen_rejected.py
        ```

        Key features:
        - Compares attention-gaze correlations between chosen and rejected responses across all models
        - Configurable analysis options (directly on the file):
            * Filter for unanimous responses only
            * Select specific reading measure to analyze
        - Results loaded from: attention/modelname/results/[chosen|rejected]/
        - Generates comparative visualizations to analyze attention differences between chosen/rejected responses

    - Plotting attention-gaze correlations for reward models:
        ```bash
        python analyse_attention/main_plot_chosen_rejected_reward.py
        ```
        Key features:
        - Compares attention-gaze correlations between chosen and rejected responses for reward models
        - Analyzes both standard attention (response only) and reward attention (prompt + response)
        - Supported reward models:
            * openbmb/UltraRM-13b
            * openbmb/Eurus-RM-7b 
            * nicolinho/QRM-Llama3.1-8B
        
        Configuration (directly on the file):
            - Filter for unanimous responses via completed/not_completed options
            - Select specific reading measures to analyze
        - Results loaded from:
            * Standard attention: attention/modelname/results/[chosen|rejected]/
            * Reward attention: attention_reward/modelname/results/[chosen|rejected]/