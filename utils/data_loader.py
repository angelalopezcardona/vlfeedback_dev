import json
import pandas as pd
from eyetrackpy.data_processor.models.eye_tracking_data_simple import (
    EyeTrackingDataUserSet,
)
import os
from PIL import Image
import numpy as np
import cv2
import re
import pathlib
#adapt all this to the new files directorys

class ETDataLoader:
    def load_gaze_features(self, folder):
        datauserset = EyeTrackingDataUserSet()
        files_responses = datauserset.search_word_coor_fixations_files(folder)
        files_prompts = ETDataLoader().search_prompt_files(folder)
        fixations_trials = {}
        for trial, file in files_responses.items():
            fixations_trials[trial] = datauserset._read_coor_trial(file)
        for trial, file in files_prompts.items():
            fixations_trials[trial] = datauserset._read_coor_trial(file)
        return fixations_trials
    
    @staticmethod
    def find_image_in_screenshot(screenshot, image, draw_result=False, out_path="match_vis.png"):
        """
        Returns (x, y, w, h) of the image found inside the screenshot, in screenshot pixel coords.
        Uses ORB feature matching + RANSAC homography (robust to scale/rotation).
        """
        scr = screenshot if isinstance(screenshot, np.ndarray) else cv2.imread(screenshot, cv2.IMREAD_COLOR)
        tpl = image if isinstance(image, np.ndarray) else cv2.imread(image, cv2.IMREAD_COLOR)
        if scr is None or tpl is None:
            raise FileNotFoundError("Could not read one of the images.")

        gray_scr = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
        gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        # 1) Detect & describe (ORB is free to use; SIFT also works if available)
        orb = cv2.ORB_create(nfeatures=5000)
        kps1, des1 = orb.detectAndCompute(gray_tpl, None)   # template (the smaller image)
        kps2, des2 = orb.detectAndCompute(gray_scr, None)   # screenshot

        if des1 is None or des2 is None:
            raise RuntimeError("Could not compute descriptors. Try increasing nfeatures or using SIFT.")

        # 2) Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # 3) Lowe’s ratio test to keep good matches
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 8:
            raise RuntimeError(f"Not enough good matches: {len(good)}. Try relaxing the ratio or using SIFT.")

        # 4) Compute homography with RANSAC
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # template points
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # screenshot points

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Homography failed. Not enough inliers or bad matches.")

        # 5) Project template corners into screenshot coords
        h, w = gray_tpl.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).reshape(4, 2)

        # 6) Get axis-aligned bounding box
        xs, ys = projected[:, 0], projected[:, 1]
        x_min, y_min = int(np.floor(xs.min())), int(np.floor(ys.min()))
        x_max, y_max = int(np.ceil(xs.max())),  int(np.ceil(ys.max()))
        # Normalize coordinates between 0 and 1 by dividing by screenshot dimensions
        h_scr, w_scr = scr.shape[:2]
        bbox = (x_min/w_scr, y_min/h_scr, (x_max-x_min)/w_scr, (y_max-y_min)/h_scr)  # (x, y, w, h) normalized

        if draw_result:
            vis = scr.copy()
            cv2.polylines(vis, [projected.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.imwrite(out_path, vis)

        return bbox

    @staticmethod
    def search_prompt_files(folder):
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        file_paths = list(folder.rglob("*"))
        files = {}
        pattern = r"word_cor_image_fixations_(\d+)\."
        for file in file_paths:
            # The regex pattern
            # Perform the match
            match = re.search(pattern, str(file))
            # Check if there is a match and extract the number
            if match:
                trial = match.group(1)
                files[trial] = file

        return files
    
    @staticmethod
    def load_prompt_screenshot(folder, prompt_number, raw_data_path = None, method='cv2'):
        image_path = folder + "user_" + str(1) + "_session_1_prompt_" + str(prompt_number) + ".png"
        if raw_data_path is not None:
            image_path = raw_data_path + image_path
        return ETDataLoader()._load_image(image_path, method=method)

    @staticmethod
    def _load_prompt_response(prompt_number, path = "/responses_files/", raw_data_path = None):
        if raw_data_path is not None:
            path = raw_data_path + path
        texts_prompts = pd.read_excel(path + "prompt_" + str(prompt_number) + ".xlsx")
        return texts_prompts
    
    @staticmethod
    def _load_prompt_image(prompt_number, path = "/images/", raw_data_path = None, method='cv2'):
        if raw_data_path is not None:
            path = raw_data_path + path
        image_path = path + "img_prompt_" + str(prompt_number) + ".jpg"

        return ETDataLoader()._load_image(image_path, method)
    

    @staticmethod
    def _load_image(image_path, method="pil"):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"JPG image file not found: {image_path}")
        
        if method == "pil":
            # Method 1: Using PIL + numpy
            image = Image.open(image_path)
            image_array = np.array(image)
            return image_array
            
        elif method == "cv2":
            # Method 2: Using OpenCV
            # Note: OpenCV loads images in BGR format by default
            image_array = cv2.imread(image_path)
            # Convert BGR to RGB if needed
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            return image_array
            
        elif method == "direct":
            # Method 3: Direct numpy load (for .npy files)
            # This would be used if you have pre-saved numpy arrays
            npy_path = image_path.replace('.jpg', '.npy')
            if os.path.exists(npy_path):
                return np.load(npy_path)
            else:
                raise FileNotFoundError(f"NPY file not found: {npy_path}")
        else:
            raise ValueError("Method must be 'pil', 'cv2', or 'direct'")
        
    @staticmethod
    def _load_prompt_image_as_array(prompt_number, path = "/images/", raw_data_path = None, method="pil"):
        """
        Load image as numpy array using different methods.
        
        Args:
            prompt_number: The prompt number
            path: Path to images directory
            parent_path: Parent directory path
            method: Method to use - "pil", "cv2", or "direct"
        
        Returns:
            numpy array of the image
        """
        if raw_data_path is not None:
            path = raw_data_path + path
        image_path = path + "img_prompt_" + str(prompt_number) + ".jpg"
        return ETDataLoader()._load_image(image_path, method)
    
    @staticmethod
    def load_prompts(path="/prompts_file/", raw_data_path = None):
        if raw_data_path is not None:
            path = raw_data_path + path
        texts_prompts = pd.read_excel(path + "prompts.xlsx")
        return texts_prompts
    

    def filter_rescale_fixations(self, data, prompt_number, box_image):
        """
        Filter fixations and rescale to cropped image dimensions
        
        Args:
            data: DataFrame with fixation data
            prompt_number: The prompt/trial number
            box_images: Dict with bbox info [x, y, w, h] in original image coordinates (0-1 range)
        
        Returns:
            Rescaled fixations in cropped image coordinates (0-1 range)
        """
        # Get fixations for this prompt
        fixations = data[data['USER']==str(prompt_number)][['x','y']].copy()

        bbox_x, bbox_y, bbox_w, bbox_h = box_image
        

        mask = (
            (fixations['x'] >= bbox_x) & 
            (fixations['x'] <= bbox_x + bbox_w) &
            (fixations['y'] >= bbox_y) & 
            (fixations['y'] <= bbox_y + bbox_h)
        )
        fixations = fixations[mask].copy()
        
        fixations['x'] = (fixations['x'] - bbox_x) / bbox_w
        fixations['y'] = (fixations['y'] - bbox_y) / bbox_h
        
        return fixations
    
    def load_data(self, raw_data_path = None, image_method="cv2"):
        responses = {}
        images = {}
        prompts_screenshots = {}
        prompts_df = self.load_prompts(path="/prompts_file/", raw_data_path=raw_data_path)    
        prompts = {row['n_prompt']: row['prompt_text'] for _, row in prompts_df.iterrows()}
        for prompt_number, _ in prompts.items():
            responses = self._load_prompt_response(prompt_number, path="/responses_files/", raw_data_path=raw_data_path)
            for _ , row in responses.iterrows():
                responses[row['n_resp']] = row["resp_text"]
            images[prompt_number] = self._load_prompt_image_as_array(
                prompt_number, path="/images/", raw_data_path=raw_data_path, method=image_method
            )
            folder = raw_data_path + "/fixations/participant_" + str(1) + "_" + str(1) + "/session_1/vertices/"
            prompts_screenshots[prompt_number] = self.load_prompt_screenshot(folder, prompt_number, method='cv2')
            
        return responses, images, prompts, prompts_screenshots
    
    def load_subject_saliency(self, subjects:list[int], raw_data_path, processed_data_path = None):
        prompts_df = self.load_prompts(path="/prompts_file/", raw_data_path = raw_data_path)    
        prompts = {row['n_prompt']: row['prompt_text'] for _, row in prompts_df.iterrows()}
        subjects_data = {}
        for subject in subjects:
            folder = processed_data_path + "/saliency/participant_" + str(subject) + "/"
            subjects_data[subject] = {}
            for prompt_number, _ in prompts.items():
                try:
                    data = np.load(folder + "saliency_{}.npy".format(str(prompt_number)))
                except:
                    print("Saliency map not found for subject {} and prompt {}".format(subject, prompt_number))
                    continue
                subjects_data[subject][prompt_number] = data
        return subjects_data
    
    def load_subject_word_data(self, subjects:list[int], raw_data_path = None):
        subjects_data = {}
        for subject in subjects:
            folder = raw_data_path + "/fixations/participant_" + str(subject) + "_" + str(subject) + "/session_1/"
            data = self.load_gaze_features(folder)
            subjects_data[subject] = data

        return subjects_data
    
    def load_subject_fixations(self, subjects:list[int], raw_data_path = None):
        subjects_data = {}
        for subject in subjects:
            folder = raw_data_path + "/fixations/participant_" + str(subject) + "_" + str(subject) + "/session_1/"
            data = pd.read_csv(folder + "participant_{}_eye_tracking_tobii_fixations.csv".format(str(subject)))
            subjects_data[subject] = data

        return subjects_data
    
    def load_texts_responses(self, raw_data_path=None):
        folder = raw_data_path + "/fixations/participant_" + str(1) + "_" + str(1) + "/session_1/"
        data = self.load_gaze_features(folder)
        return {trial: list(fixations_trial.text) for trial, fixations_trial in data.items()}


