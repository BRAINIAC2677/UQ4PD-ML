import os

DATA_BASE_PATH = os.getcwd()+"/../../../"
BASE_DIR = DATA_BASE_PATH

FACIAL_EXPRESSIONS = {
    'smile': True,
}

MODEL_TAG = "best_auroc_baal"

FEATURES_FILE = os.path.join(DATA_BASE_PATH,"data/facial_expression_smile/facial_dataset.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR, "models/facial_expression_smile_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model_config.json")
SCALER_PATH = os.path.join(MODEL_BASE_PATH,"scaler/scaler.pth")