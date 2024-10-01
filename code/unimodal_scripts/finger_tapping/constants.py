import os

BASE_DIR = os.getcwd()+"/../../../"
DATA_BASE_DIR = os.path.join(BASE_DIR, "data/") 
SAVED_MODEL_DIR_NAME = 'saved_models'

MODEL_TAG = "both_hand_fusion"

FEATURES_FILE = os.path.join(DATA_BASE_DIR,"finger_tapping/features_demography_diagnosis_Nov22_2023.csv")

MODEL_PATH = os.path.join(BASE_DIR,f"saved_models/finger_model_{MODEL_TAG}","predictive_model/model.pth")
SCALER_PATH = os.path.join(BASE_DIR, f"saved_models/finger_model_{MODEL_TAG}","scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, f"saved_models/finger_model_{MODEL_TAG}","predictive_model/model_config.json")
MODEL_BASE_PATH = os.path.join(BASE_DIR,f"saved_models/finger_model_{MODEL_TAG}")