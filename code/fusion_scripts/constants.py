import os

BASE_DIR = os.getcwd()+"/../../"
DATA_BASE_DIR = os.path.join(BASE_DIR,"data")
SAVED_MODEL_DIR_NAME = 'saved_models'

FINGER_FEATURES_FILE = os.path.join(DATA_BASE_DIR,"finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
AUDIO_FEATURES_FILE = os.path.join(DATA_BASE_DIR,"quick_brown_fox/wavlm_fox_features.csv")
FACIAL_FEATURES_FILE = os.path.join(DATA_BASE_DIR, "facial_expression_smile/facial_dataset.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR, SAVED_MODEL_DIR_NAME)
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH, "uncertainty_aware_fusion/model_config.json")
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "uncertainty_aware_fusion/model.pth")

FACIAL_EXPRESSIONS = {
    'smile': True,
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion', 'fox_model_best_auroc', 'facial_expression_smile_best_auroc'],
    1: ['finger_model_both_hand_fusion', 'fox_model_best_auroc'],
    2: ['finger_model_both_hand_fusion', 'facial_expression_smile_best_auroc'],
    3: ['fox_model_best_auroc', 'facial_expression_smile_best_auroc']
}