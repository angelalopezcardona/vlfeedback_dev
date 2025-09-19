### main_tobii_classification.py
Example run:
```bash
python main_tobii_classification.py path/to/raw.csv
```

### main_generate_synthetic_gaze.py
- Class: `SyntheticGazeGenerator(version=2, remap=False)`
- In main:
  - input_path: base folder of real users, e.g. `"users/"`
  - output_path: base folder for synthetic, e.g. `"users_synthetic/"`
  - users: list of participant ids (strings)
- Call:
  - `gaze_generator.generate_synthetic_gaze_for_all_users(users, input_path, output_path)`

### response_extractor.py
- Class: `ResponseExtractor(response_folder, model_response_folder, users_base_path)`
  - response_folder: folder with per-user survey/export files (the `_data` subfolders)
  - model_response_folder: folder with model `.xlsx` prompt files
  - users_base_path: base `users` directory where `participant_*/*/session_1` live
- In main/example:
  - Set the three paths (absolute or relative)
  - Call `extractor.process_all_users()`

### response_statistics.py
- Class: `ResponseStatisticsProcessor(input_users_dir, input_responses_dir, output_dir)`
  - input_users_dir: base `users` directory (expects `info_extended.csv` per participant)
  - input_responses_dir: folder with model `.xlsx` prompt files
  - output_dir: where to write `trial_responses.csv`, `response_model.csv`, `response_summary.csv`
- Typical flow:
  - `processor = ResponseStatisticsProcessor(...)`
  - `processor.process_all()`

### main_prompt_response.py
- Inline config at top:
  - `users_list`: list of participant ids
  - `sessions_list`: list of session numbers
- When creating `EyeTrackingDataImage`:
  - user (from loop), session (from loop), user_set (usually 0)
  - x_screen, y_screen (screen resolution)
  - path: base users path (e.g., `"./users"`)

USAGE:
1. Tobii classification: `main_tobii_classification.py`
2. Build users data: `main_prompt_response.py`
3. Create info_extended: `response_extractor.py`
4. Aggregate response stats: `response_statistics.py`