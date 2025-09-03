    
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
    subjects = [1,2,3,4,5,6,7,8,9,10]
    prompt_number_plot = [9,10]
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et"
    save_path= cwd + '/results/et/'
    figures_path = cwd + '/results/et/figures/'
    #load saliency
    subject_saliency = ETDataLoader().load_subject_saliency(subjects = subjects, raw_data_path = raw_data_path, processed_data_path = process_data_path)
    #load fixations
    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    subject_fixations = ETDataLoader().load_subject_fixations(subjects = subjects, raw_data_path=raw_data_path)
    saliency_generator = SaliencyGenerator()

    results_path_saliency_visalformer = save_path + '/visalformer/saliency'
    results_path_saliency_mdsem = save_path + '/mdsem/salmaps_arrays'

    for prompt_number, _ in prompts.items():
        box_image = ETDataLoader().find_image_in_screenshot(prompts_screenshots[prompt_number], images[prompt_number], draw_result=False)
        saliency_map_visalformer = np.load(results_path_saliency_visalformer + '/saliency_trial_{}.npy'.format(prompt_number))
        # saliency_map_mdsem_05 = np.load(results_path_saliency_mdsem + '/img_prompt_{}_500.npy'.format(prompt_number))
        # saliency_map_mdsem_3 = np.load(results_path_saliency_mdsem + '/img_prompt_{}_3000.npy'.format(prompt_number))
        saliency_map_mdsem_5 = np.load(results_path_saliency_mdsem + '/img_prompt_{}_5000.npy'.format(prompt_number))
        saliency_map_human = []
        for subject, saliency_maps in subject_saliency.items():
            fixations_trial = subject_fixations[subject]
            if prompt_number not in saliency_maps:
                continue
            saliency_map_human.append(saliency_maps[prompt_number])
        average_saliency_map_human = np.mean(saliency_map_human, axis=0)
        figure_name_visalformer="saliency_visalformer_{}.png".format(prompt_number)
        figure_name_mdsem="saliency_mdsem_{}.png".format(prompt_number)
        figure_name_human="saliency_human_{}.png".format(prompt_number)
        saliency_generator.create_overlay_and_save_saliency_map(images[prompt_number], saliency_map_visalformer, folder=figures_path, figure_name=figure_name_visalformer)
        saliency_generator.create_overlay_and_save_saliency_map(images[prompt_number], saliency_map_mdsem_5, folder=figures_path, figure_name=figure_name_mdsem)
        saliency_generator.create_overlay_and_save_saliency_map(images[prompt_number], average_saliency_map_human, folder=figures_path, figure_name=figure_name_human)
    
    for prompt_number in prompt_number_plot:
        text = latex_three_subfigs(prompt_number)
        print(text)

                


 
    


