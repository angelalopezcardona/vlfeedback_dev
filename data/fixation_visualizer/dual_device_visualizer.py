from .visualizer import FixationVisualizer
import os
import pandas as pd
import numpy as np

class DualDeviceFixationVisualizer:
    """
    A class for visualizing fixation data from two different eye-tracking devices (Tobii and Gazepoint).
    
    This class processes fixation data from multiple users and devices. It processes the fixation data by filtering based on
    duration, and creates saliency maps with and visual overlays for each device.
    The folder structure should be:
    users/
        1/
            fixations_tobii.csv
            fixations_gazepoint.csv
        2/
            fixations_tobii.csv
            fixations_gazepoint.csv
    
         Args:
         users_folder (str): Path to the folder containing user data. Each user folder should contain
                            two CSV files: one for Tobii data and one for Gazepoint data.
         images_folder (str): Path to the folder containing the image prompts to be analyzed.
         n_prompt (int): Number of image prompts to process (default: 45).
         fixation_time (int, optional): Maximum duration in milliseconds to analyze for each image (default: 5000).
                                       If None, takes all fixations within the image area (no temporal filtering).
         info_alignment (bool, optional): If True, reads info.csv file and applies calibration correction when calibrate=True (default: False).
   
    Output:
        For each user and each image prompt, generates two visualization results:
        - results_tobii/: Contains saliency maps with and visual overlays for each image
        - results_gazepoint/: Contains saliency maps with and visual overlays for each image
    """
    def __init__(self, users_folder='users', images_folder='images', n_prompt=30, fixation_time=5000, info_alignment=False):
        self.users_folder = users_folder
        self.images_folder = images_folder
        self.n_prompt = n_prompt
        self.fixation_time = fixation_time
        self.info_alignment = info_alignment
        self.x_screen = 1280
        self.y_screen = 1024
        self.calibration_counter = 0  # Contatore per le ricalibrazioni applicate
        self.process_user_data()
        self._print_calibration_summary()

    def _compute_calibration_coordinates_trial_prompt(self, fixations_raw, trial):
        """
        Compute calibration coordinates for a prompt trial using the most recent previous 'FIX_CROSS' row.
        """
        try:
            # Trova indice della prima riga del trial corrente
            trial_rows = fixations_raw[fixations_raw["USER"] == str(trial)]
            if trial_rows.empty:
                return None, None
                
            trial_index = trial_rows.index[0]

            # Cerca la prima riga con USER == 'FIX_CROSS' prima del trial_index
            fix_cross_rows = fixations_raw.loc[:trial_index - 1]
            fix_cross_mask = fix_cross_rows["USER"] == "FIX_CROS"
            
            if not fix_cross_mask.any():
                return None, None
                
            fix_cross_index = fix_cross_rows[fix_cross_mask].last_valid_index()

            if fix_cross_index is None:
                return None, None  

            # Calcola le coordinate di calibrazione
            if "FPOGX" in fixations_raw.columns:
                calibrate_x = fixations_raw.loc[fix_cross_index]["FPOGX"] - 0.5
                calibrate_y = fixations_raw.loc[fix_cross_index]["FPOGY"] - 0.5
            else:
                calibrate_x = fixations_raw.loc[fix_cross_index]["x"] - 0.5
                calibrate_y = fixations_raw.loc[fix_cross_index]["y"] - 0.5                
            
            calibrate_x *= self.x_screen
            calibrate_y *= self.y_screen

            return calibrate_x, calibrate_y
            
        except Exception as e:
            print(f"Error computing calibration coordinates for prompt trial {trial}: {str(e)}")
            return None, None

    def process_user_data(self):
        for user_folder in os.listdir(self.users_folder):
            # Salta file che non sono cartelle e cartelle che non contengono dati utente
            if 'vertices' in user_folder or not user_folder.startswith('participant_'):
                continue
                
            user_path = os.path.join(self.users_folder, user_folder)
            if not os.path.isdir(user_path):
                continue
                
            joined_path = os.path.join(user_path, 'session_1')
            if not os.path.exists(joined_path):
                print(f"Session folder not found for {user_folder}. Skipping.")
                continue
                
            fixations_tobii, fixations_gazepoint = None, None
            info_df = None

            try:
                for f in os.listdir(joined_path):
                    if f.endswith('.csv') and 'tobii' in f.lower():
                        fixations_tobii = pd.read_csv(os.path.join(joined_path, f))
                
                # Carica il file info.csv se info_alignment è True
                if self.info_alignment:
                    info_path = os.path.join(joined_path, 'info.csv')
                    if os.path.exists(info_path):
                        info_df = pd.read_csv(info_path, sep=';')
                        print(f"Loaded info.csv for {user_folder} with {len(info_df)} trials")
                    else:
                        print(f"Warning: info.csv not found for {user_folder}, info_alignment will be ignored")
                        info_df = None
                        
            except Exception as e:
                print(f"Error reading data from {joined_path}: {str(e)}. Skipping.")
                continue

            if fixations_tobii is None and fixations_gazepoint is None:
                print(f"No Tobii or Gazepoint data in {joined_path}. Skipping.")
                continue

            self._process_images(joined_path, fixations_tobii, fixations_gazepoint, info_df, user_folder)

    def _process_images(self, joined_path, fixations_tobii, fixations_gazepoint, info_df, user_folder):
        for i in range(1, self.n_prompt + 1):
            image_path = os.path.join(self.images_folder, f'img_prompt_{i}.jpg')
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist. Skipping.")
                continue

            # Carica l'immagine per ottenere le dimensioni
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}. Skipping.")
                continue
            img_height, img_width = img.shape[:2]

            # Process Tobii if available
            if fixations_tobii is not None:
                if str(i) in fixations_tobii['USER'].values:
                    df_filtered_tobii = self._filter_fixations(fixations_tobii, i, img_width, img_height)
                    
                    # Applica la ricalibrazione se info_alignment è True e calibrate=True per questo trial
                    if self.info_alignment and info_df is not None:
                        df_filtered_tobii = self._apply_calibration_if_needed(df_filtered_tobii, i, info_df, user_folder, fixations_tobii)
                    
                    self._visualize(image_path, df_filtered_tobii, joined_path, 'results_tobii', 'tobii')
                else:
                    print(f"User {i} not found in Tobii data. Skipping.")

            # Process Gazepoint if available
            if fixations_gazepoint is not None:
                # Try to infer user column name for Gazepoint, fallback to 'USER' if present
                user_col = 'USER' if 'USER' in fixations_gazepoint.columns else None
                if user_col is None or str(i) in fixations_gazepoint.get(user_col, []):
                    df_filtered_gzp = self._filter_fixations_gzp(fixations_gazepoint, i, img_width, img_height)
                    
                    # Applica la ricalibrazione se info_alignment è True e calibrate=True per questo trial
                    if self.info_alignment and info_df is not None:
                        df_filtered_gzp = self._apply_calibration_if_needed(df_filtered_gzp, i, info_df, user_folder, fixations_gazepoint)
                    
                    self._visualize(image_path, df_filtered_gzp, joined_path, 'results_gazepoint', 'gzp')
                else:
                    print(f"User {i} not found in Gazepoint data. Skipping.")

    def _filter_fixations(self, fixations, user_id, img_width, img_height):
        # 1. Filtra per utente specifico
        df_filtered = fixations[fixations['USER'] == str(user_id)].copy()
        
        # 2. RIMUOVI SEMPRE LA PRIMA FISSAZIONE (settling/ricerca iniziale)
        if len(df_filtered) > 1:
            df_filtered = df_filtered.iloc[1:].reset_index(drop=True)
        else:
            return pd.DataFrame()  # Se c'è solo una fissazione, ritorna vuoto
        
        # 3. Applica il cropping spaziale (scarta fissazioni fuori dall'immagine)
        df_filtered = self._crop_fixations_to_image_area(df_filtered, img_width, img_height)
        
        # 4. Se non ci sono fissazioni valide, ritorna DataFrame vuoto
        if df_filtered.empty:
            return df_filtered
        
        # 5. Se fixation_time è None, prendi tutte le fissazioni dentro l'immagine
        if self.fixation_time is None:
            return df_filtered
        
        # 6. Nuovo filtro temporale: somma le fissazioni in ordine temporale fino a raggiungere fixation_time
        df_filtered = df_filtered.sort_values('recording_timestamp').reset_index(drop=True)
        
        cumulative_duration = 0
        selected_fixations = []
        
        for idx, row in df_filtered.iterrows():
            fixation_duration = row['duration']
            
            # Se aggiungendo questa fissazione superiamo il limite, tronca la durata
            if cumulative_duration + fixation_duration > self.fixation_time:
                remaining_time = self.fixation_time - cumulative_duration
                if remaining_time > 0:
                    # Crea una copia della riga con durata troncata
                    row_copy = row.copy()
                    row_copy['duration'] = remaining_time
                    selected_fixations.append(row_copy)
                break
            else:
                selected_fixations.append(row)
                cumulative_duration += fixation_duration
        
        # Crea il DataFrame finale con le fissazioni selezionate
        if selected_fixations:
            df_filtered = pd.DataFrame(selected_fixations).reset_index(drop=True)
        else:
            df_filtered = pd.DataFrame()
        
        return df_filtered

    def _crop_fixations_to_image_area(self, df, img_width, img_height):
        """
        Filtra le fissazioni per tenere solo quelle dentro l'area dell'immagine.
        Assume coordinate normalizzate (0-1) e immagine centrata.
        Converte anche le coordinate in pixel per il visualizer.
        """
        # Screen aspect ratio (16:9)
        aspect_ratio = 16/9
        
        # Half size in normalized coordinates
        half_height = 0.5 / 2
        half_width = (0.5 / 2) / aspect_ratio
        
        # Bounds in normalized coordinates
        x_center, y_center = 0.5, 0.5
        x_min = x_center - half_width
        x_max = x_center + half_width
        y_min = y_center - half_height
        y_max = y_center + half_height
        
        # Select fixations inside image area
        inside = (df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)
        df_inside = df[inside].copy()
        
        if not df_inside.empty:
            # Rescale: make coordinates relative to image
            df_inside['x'] = (df_inside['x'] - x_min) / (x_max - x_min) * img_width
            df_inside['y'] = (df_inside['y'] - y_min) / (y_max - y_min) * img_height
        
        return df_inside.reset_index(drop=True)

    def _filter_fixations_gzp(self, fixations, user_id, img_width, img_height):
        fixations = fixations.rename(
            columns={"FPOGX": "x", "FPOGY": "y", "FPOGD": "duration", "FPOGID": "ID", 'TIMETICK(f=10000000)': 'recording_timestamp'}
        )
        fixations['duration'] = fixations['duration'] * 1000
        fixations['recording_timestamp'] = fixations['recording_timestamp'] / 10000
        return self._filter_fixations(fixations, user_id, img_width, img_height)

    def _apply_calibration_if_needed(self, fixations_df, trial_id, info_df, user_folder, fixations_raw):
        """
        Applica la ricalibrazione alle fissazioni se calibrate=True per questo trial nel file info.csv
        """
        if fixations_df.empty:
            return fixations_df
            
        # Cerca il trial nel file info.csv
        # I trial possono essere stringhe che rappresentano float (es. "4.1", "18.2")
        # Converti entrambi a stringa per il confronto
        trial_info = info_df[info_df['trial'].astype(str) == str(trial_id+0.0).strip()]
        
        if trial_info.empty:
            print(f"Trial {trial_id} not found in info.csv for {user_folder}")
            return fixations_df
            
        # Controlla se calibrate=True per questo trial
        if trial_info.iloc[0]['calibrate'] == True:
            print(f"Applying calibration for trial {trial_id} in {user_folder}")
            
            try:
                calibrate_x, calibrate_y = self._compute_calibration_coordinates_trial_prompt(fixations_raw, trial_id)
                
                if calibrate_x is not None and calibrate_y is not None:
                    # Applica la correzione alle coordinate delle fissazioni
                    fixations_df = fixations_df.copy()
                    fixations_df['x'] = fixations_df['x'] - calibrate_x
                    fixations_df['y'] = fixations_df['y'] - calibrate_y
                    print(f"Applied calibration correction")
                    self.calibration_counter += 1 # Incrementa il contatore
                else:
                    print(f"Could not compute calibration coordinates for trial {trial_id}")
                    
            except Exception as e:
                print(f"Error applying calibration for trial {trial_id}: {str(e)}")
                return fixations_df
        else:
            print(f"No calibration needed for trial {trial_id} in {user_folder}")
            
        return fixations_df

    def _visualize(self, image_path, fixation_df, output_path, results_folder, label):
        try:
            print(f"Result for {image_path} with {label}:")
            
            # Crea il path di output normale
            output_path_normal = os.path.join(output_path, results_folder)
            
            visualizer = FixationVisualizer(
                image_path=image_path,
                fixation_df=fixation_df,
                output_path=output_path_normal,
                mode='saliency',
                sigma=20,
                alpha=0.3,
                normalized=True,
                already_cropped=True,  # Il cropping è già stato fatto nel filtro
                fixation_time=self.fixation_time  # Passa il fixation_time per calcolare il suffisso internamente
            )
        except Exception as e:
            print(f"Error processing {image_path} with {label}: {str(e)}")

    def _print_calibration_summary(self):
        """
        Stampa il riepilogo delle ricalibrazioni applicate
        """
        if self.info_alignment:
            print("\n" + "="*60)
            print("📊 RIEPILOGO RICALIBRAZIONI")
            print("="*60)
            print(f"✅ Numero totale di ricalibrazioni applicate: {self.calibration_counter}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("📊 RIEPILOGO PROCESSAMENTO")
            print("="*60)
            print("ℹ️  Ricalibrazione non attivata (info_alignment=False)")
            print("="*60)




