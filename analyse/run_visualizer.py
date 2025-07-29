#!/usr/bin/env python3
from fixation_visualizer.dual_device_visualizer import DualDeviceFixationVisualizer
from fixation_visualizer.saliency_comparison import SaliencyEvaluator

def main():
    print("Running dual device visualizer...")
    DualDeviceFixationVisualizer(users_folder='users', images_folder='images')
    print("All users processed.")
    '''
    print("\nRunning saliency evaluator...")
    SaliencyEvaluator(
        user_root="users",
        synthetic_salmaps_root="salmaps",
        original_images_root="images",
        output_dir="salmaps_vis_avg",
        n_prompts=45
    )
    print("Saliency evaluation completed.")
    '''
if __name__ == "__main__":
    main()
