    
    # Attention computation
    ## Compute attention of each model:
    - In this folder we can compute the attention of different transformer models and compare it with the eye-tracking data.

    ### Computing attention per models and per layer in text models:
        run `python analyse_attention/main_compute_attention_text.py` to compute attention patterns across all models and layers. This file is used to compute the attention of text models, for example BERT, RoBERTa, Llama, etc.

        Key features:
        - Computes attention scores for each trial and layer across all configured models

        Configuration:
        - Models can be enabled/disabled in the models dictionary in main_compute_attention.py
        - Results are saved per layer as CSV files in:
            * Standard attention: attention/model_name/results/set_n/trial_XX/layer_X.csv
            * Reward attention: attention_reward/model_name/results/set_n/trial_XX/layer_X.csv

        Example usage:
        ```bash
        # Standard attention computation
        python analyse_attention/main_compute_attention_text.py

        ```
    ### Computing attention per models and per layer in visual models:
        run `python analyse_attention/main_compute_attention_visual.py` to compute attention patterns across all models and layers. This file is used to compute the attention of visual models, for example Llava.
        It has the possibility to compute the attention using the rollout method or the "normal" method.
        It will compute the attention for the images and the text and save the results for the images and the text. For the image in the form of saliency map and for the text in the form of a dataframe per layer, token. 

        Key features:
        - Computes attention scores for each trial and layer across all configured models
        - Can compute the attention using the rollout method or the "normal" method
        - Saves the results for the images and the text. For the image in the form of saliency map and for the text in the form of a dataframe per layer, token. 
        
        Configuration:
        - Models can be enabled/disabled in the models dictionary in main_compute_attention_visual.py
        - Results are saved per layer as CSV files in:
            * Standard attention: attention/model_name/results/set_n/trial_XX/layer_X.csv
            * Reward attention: attention_reward/model_name/results/set_n/trial_XX/layer_X.csv
        
        Example usage:
        ```bash
        # Standard attention computation
        python analyse_attention/main_compute_attention_visual.py
        ```

       ## Comparing attention with reading measures:
        ### Comparing attention with reading measures in text models:
        This script analyzes correlations between model attention patterns and human reading measures.

        Usage:
        ```bash
        # Basic usage
        python analyse_attention/main_compute_compare_trials_text.py

        # Filter for unanimous responses only
        python analyse_attention/main_compute_compare_trials_text.py 

        

        Key features:
        - Computes correlations between model attention and reading measures (fixation duration, count etc.)

        - Saves results separately for chosen and rejected responses
        - Results saved to:
            * Standard attention: attention/modelname/results/[chosen|rejected]/

        Configuration:
        - Models can be enabled/disabled in the models dictionary in the script
        - Available reading measure can be configured in the gaze_features list

        ### Comparing attention with reading measures in visual models:
        This script analyzes correlations between model attention patterns and human reading measures.

        Usage:
        ```bash
        # Basic usage
        python analyse_attention/main_compute_compare_trials_visual.py

        # Filter for unanimous responses only
        python analyse_attention/main_compute_compare_trials_visual.py 

    ## Plotting attention-gaze correlations:
    ### Plotting attention-gaze correlations by model layer:
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

    ### Plotting chosen vs rejected attention correlations in responses:
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

    ### Plotting visual attention-gaze correlations:
        ```bash
        python analyse_attention/main_plot_attention_visual.py
        ```

