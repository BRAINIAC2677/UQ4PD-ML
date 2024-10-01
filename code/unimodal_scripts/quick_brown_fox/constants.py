import os

BASE_DIR = os.getcwd()+"/../../../"
DATA_BASE_DIR = os.path.join(BASE_DIR, "data/")
SAVED_MODEL_DIR_NAME = 'saved_models'

MODEL_TAG = "best_auroc"

CLASSICAL_FEATURES_FILE = os.path.join(DATA_BASE_DIR,"quick_brown_fox/classical_fox_features.csv")
IMAGEBIND_FEATURES_FILE = os.path.join(DATA_BASE_DIR,"quick_brown_fox/imagebind_fox_features.csv")
WAV2VEC_FEATURES_FILE = os.path.join(DATA_BASE_DIR,"quick_brown_fox/wav2vec_fox_features.csv")
WAVLM_FEATURES_FILE = os.path.join(DATA_BASE_DIR,"quick_brown_fox/wavlm_fox_features.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR, f"{SAVED_MODEL_DIR_NAME}/fox_model_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model.pth")
SCALER_PATH = os.path.join(MODEL_BASE_PATH,"scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model_config.json")