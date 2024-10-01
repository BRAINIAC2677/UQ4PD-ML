import os

BASE_DIR = os.getcwd()+"/../../../"
DATA_BASE_DIR = os.path.join(BASE_DIR, "data/") 
SAVED_MODEL_DIR_NAME = 'saved_models'

FACIAL_EXPRESSIONS = {
    'smile': True,
}

MODEL_TAG = "best_auroc"
FEATURES_FILE = os.path.join(DATA_BASE_DIR,"facial_expression_smile/facial_dataset.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR, f"{SAVED_MODEL_DIR_NAME}/facial_expression_smile_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model_config.json")
SCALER_PATH = os.path.join(MODEL_BASE_PATH,"scaler/scaler.pth")