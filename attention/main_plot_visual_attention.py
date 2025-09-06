    
import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
from utils.data_loader import ETDataLoader
import numpy as np
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator
from eyetrackpy.data_generator.utils.saliency_metrics import compute_cc, compute_kl, compute_nss, compute_auc, compute_sim
import pandas as pd

def latex_three_subfigs(prompt_number):              
    figure_name_visalformer="saliency_visalformer_{}.png".format(prompt_number)
    figure_name_mdsem="saliency_mdsem_{}.png".format(prompt_number)
    figure_name_human="saliency_human_{}.png".format(prompt_number)
    figures_path = 'figures/saliency/'
    text = _print_latex_three_subfigs(figures_path + figure_name_human, figures_path + figure_name_mdsem, figures_path + figure_name_visalformer, cap1="Human", cap2="MDSEM", cap3="Visalformer", overall_caption="Saliency map for prompt {}".format(prompt_number), label="fig:saliency_map_{}".format(prompt_number))
    return text

def _print_latex_three_subfigs(fig1, fig2, fig3,
                        cap1="Caption 1", cap2="Caption 2", cap3="Caption 3",
                        overall_caption="Overall caption for the three panels.",
                        label="fig:three_horiz"):
    """
    Return LaTeX code for three horizontal subfigures using `subcaption`.
    Usage:
        print(latex_three_subfigs("figures/a.png","figures/b.png","figures/c.png"))
    """
    return f"""


\\begin{{figure}}[htbp]
    \\centering
    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{{fig1}}}
        \\caption{{{cap1}}}
        \\label{{fig:sub1}}
    \\end{{subfigure}}\\hfill
    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{{fig2}}}
        \\caption{{{cap2}}}
        \\label{{fig:sub2}}
    \\end{{subfigure}}\\hfill
    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{{fig3}}}
        \\caption{{{cap3}}}
        \\label{{fig:sub3}}
    \\end{{subfigure}}
    \\caption{{{overall_caption}}}
    \\label{{{label}}}
\\end{{figure}}
""".strip()




if __name__ == "__main__":
    prompt_number_plot = list(range(1, 31))
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = cwd + "/data/raw/et"
    # process_data_path = cwd + "/results/et/attention_rollout/llava-1.5-7b-hf/saliency/"
    process_data_path = cwd + "/attention_saliency/llava7b_attention/"
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    saliency_generator = SaliencyGenerator()

    for prompt_number, _ in prompts.items():
        # saliency_map = np.load(process_data_path + 'trial_{}/saliency_trial_{}_layer_gen_rollout.npy'.format(prompt_number, prompt_number))
        saliency_map = np.load(process_data_path + 'img_prompt_{}_rollout_max.npy'.format(prompt_number))


        figure_name_human="saliency_human_{}.png".format(prompt_number)
        saliency_generator.create_overlay_and_save_saliency_map(
            images[prompt_number], 
            saliency_map, 
            folder=process_data_path , 
            figure_name='saliency_second_trial_{}.png'.format(prompt_number)
            )

 
                
# /data/alop/mllm_study/results/et/attention_rollout/llava-1.5-7b-hf/saliency/trial_1/saliency_trial_1_layer_gen_rollout.npy


 
    


