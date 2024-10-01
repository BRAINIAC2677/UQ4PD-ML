import os
import copy
import pickle
import re
import math
import json
import random
import click
import scipy
import wandb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp

from baal.modelwrapper import ModelWrapper

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTEN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from constants import *
from models import *


'''
Find the GPU that has max free space
'''
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

'''
set-up device (for gpu support)
'''
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Running on {device} ...")

if device == 'cuda':
    results = get_gpu_memory()
    gpu_id = np.argmax(results)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

'''
creating necessary dirs for saving the trained models
'''
if not os.path.exists(MODEL_BASE_PATH):
    os.mkdir(MODEL_BASE_PATH)
    os.mkdir(os.path.join(MODEL_BASE_PATH,"uncertainty_aware_fusion"))

if not os.path.exists(os.path.join(MODEL_BASE_PATH,"uncertainty_aware_fusion")):
    os.mkdir(os.path.join(MODEL_BASE_PATH,"uncertainty_aware_fusion"))

#1. Load dev and test sets (participant ids)
with open(os.path.join(DATA_BASE_DIR,"dev_set_participants.txt")) as f:
    ids = f.readlines()
    dev_ids = set([x.strip() for x in ids])

with open(os.path.join(DATA_BASE_DIR,"test_set_participants.txt")) as f:
    ids = f.readlines()
    test_ids = set([x.strip() for x in ids])
	
print(f"Number of patients in the dev and test set: {len(dev_ids)}, {len(test_ids)}")


#2. process the datasets
'''
Parse date from filenames. 
Some examples:
    2022-03-24T13%3A32%3A36.977Z_NIHNT179KNNF4_finger_tapping_left.mp4 -- 2022-03-24
    2021-08-30T20%3A00%3A03.162Z_ZTi20lXEMSdqXLxtnTotwoyADq03_finger_tapping_left.mp4 -- 2021-08-30
    NIHYM875FLXFF-finger_tapping-2021-03-17T18-13-01-902Z-.mp4 -- 2021-03-17
    2019-10-21T22-16-00-772Z35-finger_tapping.mp4 -- 2019-10-21
'''
def parse_date(name:str):
    match = re.search(r"\d{4}-\d{2}-\d{2}", name)
    date = match.group()
    return date


def load_smile_data(drop_correlated = True, corr_thr = 0.85):
    df = pd.read_csv(FACIAL_FEATURES_FILE)

    #Fill data point by 0 if it is null
    df.fillna(0, inplace=True)
    
    '''
    Get the expression relavant feature columns and the feature dataframe
    '''
    feature_columns = []
    for feature in df.columns:
        for expression in FACIAL_EXPRESSIONS.keys():
            if FACIAL_EXPRESSIONS[expression] and expression in feature.lower():
                feature_columns.append(feature)
                break
    df_features = df[feature_columns]

    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if drop_correlated:
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                if val >= corr_thr:
                    drop_cols.append(col.values[0])

        drops = set(drop_cols)
        
        # Drop features from both the main and the feature dataframe
        df.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)
    # end of drop correlated columns implementation
    
    features = df.loc[:, df_features.columns[0]:df_features.columns[-1]]
    columns = df_features.columns
    features = features.to_numpy()

    df["id"] = df['ID']
    df["date"] = df.Filename.apply(parse_date)
    df["id_date"] = df["id"]+"#"+df["date"]
    df["label"] = 1.0*(df["pd"]!="no")

    return features, df["label"], df["id"], columns, df["id_date"]


def load_qbf_data(drop_correlated = False, corr_thr = 0.85, feature_files=[AUDIO_FEATURES_FILE]):
    def parse_patient_id(name:str):
        if name.startswith("NIH"): [ID, *_] = name.split("-")
        elif name.endswith("-quick_brown_fox.mp4"): [*_, ID, _] = name.split("-")
        elif name.endswith("_quick_brown_fox.mp4"): [_, ID, _, _, _] = name.split("_")
        else: [*_, ID, _, _, _] = name.split("_")
        return ID
    
    dataframes = []
    for FEATURES_FILE in feature_files:
        df_temp = pd.read_csv(FEATURES_FILE)
        dataframes.append(df_temp)

    assert (len(dataframes)>=1) and (len(dataframes)<=2)
    df = dataframes[0]
    df = df.copy()
    #print(df.columns[:20]) #'Filename', 'Participant_ID', 'gender', 'age', 'race', 'pd', f'wavlm_feature{x}'
    for i in range(1,len(feature_files)):
        df = pd.merge(left=df, right=dataframes[i], how='inner', on='Filename')

    if len(dataframes)==2:
        df = df.drop(columns=['Participant_ID_y', 'gender_y', 'age_y', 'race_y', 'pd_y'])
        df = df.rename(columns={'Participant_ID_x':'Participant_ID', 'gender_x':'gender', 'age_x':'age', 'race_x':'race', 'pd_x':'pd'})

    '''
    Drop data point if any of the feature is null
    '''
    df = df.dropna(subset = df.columns.difference(['Filename','Participant_ID', 'gender','age','race']), how='any')
 
    #Drop metadata columns to focus on features
    df_features = df.drop(columns=['Filename','Participant_ID', 'gender','age','race','pd'])
 
    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if drop_correlated:
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []
 
        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)
 
                if val >= corr_thr:
                    drop_cols.append(col.values[0])
 
        drops = set(drop_cols)
 
        # Drop features from both the main and the feature dataframe
        df.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)
    # end of drop correlated columns implementation
        
    features = df.loc[:, df_features.columns[0]:df_features.columns[-1]]
    columns = features.columns
    features = features.to_numpy()
    
    df["id"] = df.Filename.apply(parse_patient_id)
    df["date"] = df.Filename.apply(parse_date)
    df["id_date"] = df["id"]+"#"+df["date"]
    df["label"] = df["pd"]
    return features, df["label"], df["id"], columns, df["id_date"]


def load_finger_data(hand="left",drop_correlated = False, corr_thr = 0.85):
    '''
    Parse ID from filenames. 
    Some examples:
        2022-03-24T13%3A32%3A36.977Z_NIHNT179KNNF4_finger_tapping_left.mp4 -- NIHNT179KNNF4
        2021-08-30T20%3A00%3A03.162Z_ZTi20lXEMSdqXLxtnTotwoyADq03_finger_tapping_left.mp4 -- ZTi20lXEMSdqXLxtnTotwoyADq03
        NIHYM875FLXFF-finger_tapping-2021-03-17T18-13-01-902Z-.mp4 -- NIHYM875FLXFF
        2019-10-21T22-16-00-772Z35-finger_tapping.mp4 -- 772Z35
    '''
    def parse_patient_id(name:str):
        if name.startswith("NIH"): [ID, *_] = name.split("-")
        elif name.endswith("finger_tapping.mp4"): [*_, ID, _] = name.split("-")
        else: [*_, ID, _, _, _] = name.split("_")
        return ID

    df = pd.read_csv(FINGER_FEATURES_FILE)

    #Drop data point if any of the feature is null
    df = df.dropna(subset = df.columns.difference(['Unnamed: 0','filename','Protocol','Participant_ID','Task',
                'Duration','FPS','Frame_Height','Frame_Width','gender','age','race',
                'ethnicity']), how='any')
    
    '''
    Restrict only to one hand (if specified)
    '''    
    if hand!="both" and hand in ["left","right"]:
        df = df[df["hand"]==hand]

    #Drop metadata columns to focus on features
    df_features = df.drop(columns=['Unnamed: 0','filename','Protocol','Participant_ID','Task',
                'Duration','FPS','Frame_Height','Frame_Width','gender','age','race',
                'ethnicity','pd','hand'])
    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if drop_correlated:
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                if val >= corr_thr:
                    drop_cols.append(col.values[0])

        drops = set(drop_cols)
        
        # Drop features from both the main and the feature dataframe
        df.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)
    # end of drop correlated columns implementation
    
    features = df.loc[:, df_features.columns[0]:df_features.columns[-1]]
    columns = features.columns
    features = features.to_numpy()
    
    df["id"] = df.filename.apply(parse_patient_id)
    df["date"] = df.filename.apply(parse_date)
    df["id_date"] = df["id"]+"#"+df["date"]
    df["label"] = 1.0*(df["pd"]!="no")

    return features, df["label"], df["id"], columns, df["id_date"]


'''
Split the dataframe into train(+dev) and test sets
'''
def train_test_split(df):
    train_df = df[~df["id"].isin(test_ids)]
    test_df = df[df["id"].isin(test_ids)]
    return train_df, test_df


'''
Randomly split the train set into train and validation
dev_size is a ratio, i.e., 0.20 would mean 80% train - 20% dev
'''
def train_dev_split(train_df, dev_size=0.20):
    dev_df = train_df[train_df["id"].isin(dev_ids)]
    train_df = train_df[~train_df["id"].isin(dev_ids)]
    return train_df, dev_df


'''
Given a dataframe, perform oversampling
input_df: must contain columns features_0, features_1, ..., features_(N-1), and label
output_df: oversamples the minority class and returns in similar format
other columns (i.e., id, filename, etc.) will be removed.
'''
def concat_features(row):
    return np.concatenate([row[f"features_{i}"] for i in range(NUM_MODELS)])


def concat_finger_features(row):
    return np.concatenate([row[f"features_right"], row[f"features_left"]])


def oversample(input_df, sampler):
    feature_shapes = [input_df.iloc[0][f"features_{i}"].shape[0] for i in range(NUM_MODELS)]
    input_df["concat_features"] = input_df.apply(concat_features, axis=1)
    features = input_df.loc[:, "concat_features"]
    labels = input_df.loc[:,"label"]

    X = np.asarray([features.iloc[i] for i in range(len(features))])
    Y = np.asarray([labels.iloc[i] for i in range(len(labels))])

    X, Y = sampler.fit_resample(X, Y)
    output_data = []
    for (x,y) in zip(X,Y):
        data = {}
        start_index = 0
        for i in range(len(feature_shapes)):
            end_index = start_index + feature_shapes[i]
            data[f"features_{i}"] = x[start_index:end_index]
            start_index = end_index

        data["label"] = y
        output_data.append(data)

    output_df = pd.DataFrame.from_dict(output_data)
    return output_df


NUM_MODELS = 0 #just initiate here, later updated based on config


'''
Pytorch Dataset class
'''
class TensorDataset(Dataset):
    def __init__(self,df):
        '''
        df.columns: features_0, features_1, features_2, label, ...
        '''
        self.features = []
        for i in range(NUM_MODELS):
            f = torch.Tensor(np.asarray(df[f"features_{i}"].tolist()))
            self.features.append(f)

        self.labels = torch.Tensor(np.asarray(df["label"]))

    def __getitem__(self, index):
        features = []
        for i in range(NUM_MODELS):
            features.append(self.features[i][index])
        return features, self.labels[index]

    def __len__(self):
        return len(self.labels)


'''
Evaluate performance on validation/test set.
Returns all the metrics defined above and the loss.
'''
def expected_calibration_error(y, y_pred_scores, num_buckets=20):
    y_pred_scores = np.asarray(y_pred_scores).flatten()
    
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, num_buckets + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.maximum(y_pred_scores, 1.0-y_pred_scores)

    # get predictions from confidences (positional in this case)
    predicted_label = (y_pred_scores>=0.5)
    
    # get a boolean list of correct/false predictions
    accuracies = (predicted_label==y)

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


'''
Given labels and prediction scores, make a comprehensive evaluation. 
i.e., threshold = 0.5 means prediction>0.5 will be considered as positive
'''
def compute_metrics(y_true, y_pred_scores, threshold = 0.5):
    labels = np.asarray(y_true).reshape(-1)
    pred_scores = np.asarray(y_pred_scores).reshape(-1)
    preds = (pred_scores >= threshold)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['average_precision'] = average_precision_score(labels, pred_scores)
    metrics['auroc'] = roc_auc_score(labels, pred_scores)
    metrics['f1_score'] = f1_score(labels, preds)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics["confusion_matrix"] = {"tn":tn, "fp":fp, "fn":fn, "tp":tp}
    metrics["weighted_accuracy"] = (safe_divide(tp, tp + fp) + safe_divide(tn, tn + fn)) / 2.0

    '''
    True positive rate or recall or sensitivity: probability of identifying a positive case 
    (often called the power of a test)
    '''
    metrics['TPR'] = metrics['recall'] = metrics['sensitivity'] = recall_score(labels, preds)
    
    '''
    False positive rate: probability of falsely identifying someone as positive, who is actually negative
    '''
    metrics['FPR'] = safe_divide(fp, fp+tn)
    
    '''
    Positive Predictive Value: probability that a patient with a positive test result 
    actually has the disease
    '''
    metrics['PPV'] = metrics['precision'] = precision_score(labels, preds)
    
    '''
    Negative predictive value: probability that a patient with a negative test result 
    actually does not have the disease
    '''
    metrics['NPV'] = safe_divide(tn, tn+fn)
    
    '''
    True negative rate or specificity: probability of a negative test result, 
    conditioned on the individual truly being negative
    '''
    metrics['TNR'] = metrics['specificity'] = safe_divide(tn,(tn+fp))

    '''
    Brier score
    '''
    metrics['BS'] = brier_score_loss(labels, pred_scores)

    '''
    Expected Calibration Error
    '''
    metrics['ECE'] = expected_calibration_error(labels, pred_scores)
    
    return metrics


'''
Main evaluation loop to test the fusion model
'''
def evaluate_fusion_model(fusion_model, dataloader, prediction_models, config, split="dev"):
    fusion_model.eval()
    z_critical = scipy.stats.t.ppf(q=0.975, df = config["num_trials"]-1)

    all_labels = [] #true labels
    all_pred_scores = [[] for i in range(NUM_MODELS)] #unimodal prediction scores
    all_final_predictions = [] #fusion predictions
    uncertain_indices = [] #indices where the 95% CI contains 0.5
    loss = 0 #average loss
    n_samples = 0 #number of examples in the dataloader

    criterion = torch.nn.BCELoss() #loss function
    wrapped_prediction_models = [ModelWrapper(prediction_models[i],criterion) for i in range(NUM_MODELS)]
    fusion_model.eval()
    wrapped_fusion_model = ModelWrapper(fusion_model, criterion)
    
    for idx, batch in enumerate(dataloader):
        x = [[] for i in range(NUM_MODELS)] #[x0, x1, ..., xn]
        y_pred_scores = [[] for i in range(NUM_MODELS)] #probs[y0, y1, ..., yn]
        y_preds = [[] for i in range(NUM_MODELS)] #binary[y0, y1, ..., yn]
        y_vars = [[] for i in range(NUM_MODELS)]
        
        (x, y) = batch
        y = y.to(device)

        for i in range(NUM_MODELS):
            x[i] = x[i].to(device)

            if (split!="test") and (config["validation_random_noise"]=="yes"):
                noise = torch.randn(x[i].shape).to(device)
                adjusted_noise = noise*config["noise_variance"]
                x[i] += adjusted_noise

            y_multi_preds = wrapped_prediction_models[i].predict_on_batch(x[i], iterations=config["num_trials"])
            y_pred_scores[i] = y_multi_preds.mean(dim=-1).reshape(-1)
            y_vars[i] = y_multi_preds.std(dim=-1).reshape(-1)
            y_preds[i] = (y_pred_scores[i]>=0.5)
            all_pred_scores[i].extend(y_pred_scores[i].to('cpu').numpy())

        all_labels.extend(y.to('cpu').numpy())
        
        #forward pass
        with torch.no_grad():
            final_pred_scores = wrapped_fusion_model.predict_on_batch((x, y_pred_scores, y_vars), iterations=config["num_trials"])
            standard_error = (z_critical*final_pred_scores.std(dim=-1).reshape(-1))/math.sqrt(len(final_pred_scores))
            final_pred_scores = final_pred_scores.mean(dim=-1).reshape(-1)
            index_mask = (final_pred_scores-standard_error<=0.50) & (final_pred_scores+standard_error>=0.50)
            n = final_pred_scores.shape[0]
            loss += criterion(final_pred_scores.reshape(-1), y)*n
            n_samples+=n
        
        all_final_predictions.extend(final_pred_scores.cpu().numpy())
        uncertain_indices.extend(index_mask.cpu().numpy())

    #evaluate
    uncertain_indices = np.asarray(uncertain_indices).flatten()
    all_labels = np.asarray(all_labels).flatten()
    all_final_predictions = np.asarray(all_final_predictions).flatten()
    
    if split=="test":
        coverage = (len(all_labels) - uncertain_indices.sum())/len(all_labels)
        all_labels = all_labels[~uncertain_indices]
        all_final_predictions = all_final_predictions[~uncertain_indices]

    metrics = compute_metrics(all_labels, all_final_predictions)
    metrics["loss"] = loss.to('cpu').item() / n_samples
    if split=="test":
        metrics['coverage'] = coverage

    return metrics


@click.command()
@click.option("--fusion_model", default="bayesian", help="Options: 'base', 'bayesian'")
@click.option("--wandb", default="False", help="Options: 'True', 'False'")
@click.option("--learning_rate", default=0.020724443604128343, help="Learning rate for classifier")
@click.option("--dropout_prob", default=0.495989214406461)
@click.option("--num_buckets", default=20, help="Options: 5, 10, 20, 50, 100")
@click.option("--num_trials", default=30, help="Options: 100-1000")
@click.option("--uncertainty_weight", default=81.81790352752515)
@click.option("--minority_oversample",default='no',help="Options: 'yes', 'no'")
@click.option("--sampler", default='SMOTE', help="Options:SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTEN, RandomOverSampler, SMOTEENN, SMOTETomek")
@click.option("--train_random_noise", default="no", help="Options: yes, no")
@click.option("--validation_random_noise", default="no", help="Options: yes, no")
@click.option("--increase_variance",default="no", help="Options: yes, no")
@click.option("--temperature", default=0.05, help="Float between 0 and 1")
@click.option("--noise_variance",default=0.01,help="Float between 0 and 1")
@click.option("--random_state", default=357, help="Random state for classifier")
@click.option("--model_subset_choice", default=0, help="4 possible choices. See Constants.py")
@click.option("--seed", default=423, help="Seed for random")
@click.option("--batch_size",default=1024)
@click.option("--num_epochs",default=164)
@click.option("--hidden_dim", default=512)
@click.option("--query_dim", default=64)
@click.option("--last_hidden_dim", default=128)
@click.option("--optimizer",default="SGD",help="Options: SGD, AdamW, RMSprop")
@click.option("--beta1",default=0.9)
@click.option("--beta2",default=0.999)
@click.option("--weight_decay",default=0.0001)
@click.option("--momentum",default=0.6897821582954526)
@click.option("--use_scheduler",default='no',help="Options: yes, no")
@click.option("--scheduler",default='reduce',help="Options: step, reduce")
@click.option("--step_size",default=5)
@click.option("--gamma",default=0.57143922410234)
@click.option("--patience",default=5)
def main(**cfg):
    global NUM_MODELS

    '''
    save the configurations obtained from wandb (or command line) into the model config file
    '''
    with open(MODEL_CONFIG_PATH,"w") as f:
        f.write(json.dumps(cfg))

    if cfg["wandb"]=="True":
        wandb.init(project="uqparknet", entity="asifazad0178", config=cfg, tags=["fusion-model"])

    #reproducibility control
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    selected_models = MODEL_SUBSETS[cfg["model_subset_choice"]]
    NUM_MODELS = len(selected_models)

    '''
    Load the paths of the saved models that will be fused together
    '''
    model_paths = []
    for i in range(NUM_MODELS):
        path = {}
        MODEL_TAG = selected_models[i]
        path["PREDICTOR_CONFIG"] = os.path.join(MODEL_BASE_PATH, MODEL_TAG,"predictive_model/model_config.json")
        path["PREDICTOR_MODEL"] = os.path.join(MODEL_BASE_PATH, MODEL_TAG,"predictive_model/model.pth")
        path["SCALER"] = os.path.join(MODEL_BASE_PATH, MODEL_TAG,"scaler/scaler.pth")
        model_paths.append(path)
   
    processed_datasets = []
    for i in range(NUM_MODELS):
        predictor_config = {}
        with open(model_paths[i]["PREDICTOR_CONFIG"]) as json_file:
            predictor_config = json.load(json_file)

        drop_correlated = (predictor_config["drop_correlated"]=='yes')

        model_name = selected_models[i]

        if "finger_model" in model_name:
            features_right, labels_right, ids_right, columns, id_dates_right = load_finger_data(drop_correlated=drop_correlated, corr_thr=predictor_config["corr_thr"], hand="right")
            features_left, labels_left, ids_left, columns, id_dates_left = load_finger_data(drop_correlated=drop_correlated, corr_thr=predictor_config["corr_thr"], hand="left")
            
            df_right = pd.DataFrame.from_dict({"features_right":list(features_right), "id_right":ids_right, "row_id":id_dates_right, "label_right":labels_right})
            df_left = pd.DataFrame.from_dict({"features_left":list(features_left), "id_left":ids_left, "row_id":id_dates_left, "label_left":labels_left})
            
            df_both = pd.merge(df_right, df_left, how="inner", on="row_id")
            df_both = df_both.drop(columns=['label_left', 'id_left'])
            df_both = df_both.rename(columns={"label_right":"label", "id_right":"id"})
            df_both["features"] = df_both.apply(concat_finger_features, axis=1)
            
            features = np.stack(df_both.loc[:, "features"])
            labels = df_both.loc[:, "label"]
            ids = df_both.loc[:, "id"]
            row_ids = df_both.loc[:, "row_id"]
        elif "fox_model" in model_name:
            features, labels, ids, columns, row_ids = load_qbf_data(drop_correlated = drop_correlated, corr_thr = predictor_config["corr_thr"])
        elif "facial_expression_smile" in model_name:
            features, labels, ids, columns, row_ids = load_smile_data(drop_correlated = drop_correlated, corr_thr = predictor_config["corr_thr"])
        else:
            raise ValueError("Invalid model name")

        # scale if needed
        if predictor_config["use_feature_scaling"]=="yes":
            scaler = pickle.load(open(model_paths[i]['SCALER'],'rb'))
            features = scaler.transform(features)

        all_data = pd.DataFrame.from_dict({f"features_{i}":(list)(features), f"label_{i}":labels, f"id_{i}":ids, "row_id":row_ids})
        processed_datasets.append(all_data)

        if i>0:
            all_data = pd.merge(processed_datasets[i-1], processed_datasets[i], on="row_id")
            all_data = all_data.drop(columns=[f'label_{i}', f'id_{i}'])
            processed_datasets[i] = all_data

    df = processed_datasets[NUM_MODELS-1]
    df = df.rename(columns={"label_0":"label", "id_0":"id"})
    
    print("Data of finger tapping, audio, and smile is combined and loaded.")
    
    train_df, test_df = train_test_split(df)
    train_df, dev_df = train_dev_split(train_df)
    
    print(f"Number of training samples: {len(train_df)}. Positive class: {len(train_df[train_df['label']==1.0])}, Negative class: {len(train_df[train_df['label']==0.0])}.")
    print(f"Number of validation samples: {len(dev_df)}. Positive class: {len(dev_df[dev_df['label']==1.0])}, Negative class: {len(dev_df[dev_df['label']==0.0])}.")
    print(f"Number of test samples: {len(test_df)}. Positive class: {len(test_df[test_df['label']==1.0])}, Negative class: {len(test_df[test_df['label']==0.0])}.")
    
    if cfg["minority_oversample"]=="yes":
        if cfg["sampler"] == "SMOTE":
            sampler = SMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTENC":
            sampler = SMOTENC(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SVMSMOTE":
            sampler = SVMSMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "ADASYN":
            sampler = ADASYN(random_state = cfg['random_state'])
        elif cfg["sampler"] == "BorderlineSMOTE":
            sampler = BorderlineSMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "KMeansSMOTE":
            sampler = KMeansSMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTEN":
            sampler = SMOTEN(random_state = cfg['random_state'])
        elif cfg["sampler"] == "RandomOverSampler":
            sampler = RandomOverSampler(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTEENN":
            sampler = SMOTEENN(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTETomek":
            sampler = SMOTETomek(random_state = cfg['random_state'])
        else:
            raise ValueError("Invalid sampler")

        train_df = oversample(train_df, sampler)

    train_dataset = TensorDataset(train_df)
    dev_dataset = TensorDataset(dev_df)
    test_dataset = TensorDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'])

    features, label = train_dataset[0]
    feature_shapes = [features[i].shape[0] for i in range(NUM_MODELS)]
    
    prediction_models = []
    for i in range(NUM_MODELS):
        predictor_config = {}
        with open(model_paths[i]["PREDICTOR_CONFIG"]) as json_file:
            predictor_config = json.load(json_file)

        if predictor_config["model"] == "ShallowANN":    
            frozen_model = ShallowANN(feature_shapes[i], drop_prob=predictor_config["dropout_prob"])
            frozen_model.load_state_dict(torch.load(model_paths[i]["PREDICTOR_MODEL"], weights_only=True))
            frozen_model = frozen_model.to(device)
            frozen_model.fc.weight.requires_grad = False
            frozen_model.fc.bias.requires_grad = False

        elif predictor_config["model"] == "ShallowBNN":    
            frozen_model = ShallowBNN(feature_shapes[i], drop_prob=predictor_config["dropout_prob"])
            frozen_model.load_state_dict(torch.load(model_paths[i]["PREDICTOR_MODEL"], weights_only=True))
            frozen_model = frozen_model.to(device)
            frozen_model.fc.weight_mu.requires_grad = False  # Adjusted for BayesLinear
            frozen_model.fc.weight_sigma.requires_grad = False  # Adjusted for BayesLinear
            frozen_model.fc.bias_mu.requires_grad = False
            frozen_model.fc.bias_sigma.requires_grad = False

        elif predictor_config["model"] == "ANN":
            frozen_model = ANN(feature_shapes[i], drop_prob=predictor_config["dropout_prob"])
            frozen_model.load_state_dict(torch.load(model_paths[i]["PREDICTOR_MODEL"], weights_only=True))
            frozen_model = frozen_model.to(device)
            frozen_model.fc1.weight.requires_grad = False
            frozen_model.fc1.bias.requires_grad = False
            frozen_model.fc2.weight.requires_grad = False
            frozen_model.fc2.bias.requires_grad = False

        elif predictor_config["model"] == "BNN":
            frozen_model = BNN(feature_shapes[i], drop_prob=predictor_config["dropout_prob"])
            frozen_model.load_state_dict(torch.load(model_paths[i]["PREDICTOR_MODEL"], weights_only=True))
            frozen_model = frozen_model.to(device)
            frozen_model.fc1.weight_mu.requires_grad = False  # Adjusted for BayesLinear
            frozen_model.fc1.weight_sigma.requires_grad = False  # Adjusted for BayesLinear
            frozen_model.fc1.bias_mu.requires_grad = False
            frozen_model.fc1.bias_sigma.requires_grad = False
            frozen_model.fc2.weight_mu.requires_grad = False  # Adjusted for BayesLinear
            frozen_model.fc2.weight_sigma.requires_grad = False  # Adjusted for BayesLinear
            frozen_model.fc2.bias_mu.requires_grad = False
            frozen_model.fc2.bias_sigma.requires_grad = False

        else: 
            raise ValueError("Invalid Model")

        prediction_models.append(frozen_model)

    print("All the prediction models are loaded as frozen.")

    if cfg["fusion_model"] == "base":
        fusion_model = HybridFusionNetworkWithUncertainty(feature_shapes, cfg)
    elif cfg['fusion_model'] == "bayesian":
        fusion_model = HybridFusionNetworkWithUncertaintyBNN(feature_shapes, cfg) # todo: make a config attribute for this

    fusion_model = fusion_model.to(device)
    
    criterion = nn.BCELoss()
    if cfg["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(fusion_model.parameters(),lr=cfg['learning_rate'],betas=(cfg['beta1'],cfg['beta2']),weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(fusion_model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="RMSprop":
        optimizer = torch.optim.RMSprop(fusion_model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    else:
        raise ValueError("Invalid optimizer")

    if cfg["use_scheduler"]=="yes":
        if cfg['scheduler']=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        elif cfg['scheduler']=="reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['gamma'], patience = cfg['patience'])
        else:
            raise ValueError("Invalid scheduler")

    best_model = copy.deepcopy(fusion_model)
    best_dev_loss = np.finfo('float32').max
    best_dev_accuracy = 0
    best_dev_balanced_accuracy = 0
    best_dev_auroc = 0
    best_dev_f1 = 0
    wrapped_prediction_models = [ModelWrapper(prediction_models[i],criterion) for i in range(NUM_MODELS)]

    epoch_no = 0

    for epoch in tqdm(range(cfg['num_epochs'])):
        noise_variance = 0.0
        if (cfg["train_random_noise"] == "yes") and (cfg["increase_variance"] == "yes"):
            noise_variance = cfg["noise_variance"] * (1 - math.exp(-epoch_no * cfg["temperature"]))
            epoch_no += 1

        all_labels = []
        all_pred_scores = [[] for _ in range(NUM_MODELS)]

        for idx, batch in enumerate(train_loader):
            x = [[] for _ in range(NUM_MODELS)]
            (x, y) = batch
            y = y.to(device)
            y_pred_scores = [[] for _ in range(NUM_MODELS)]
            y_preds = [[] for _ in range(NUM_MODELS)]
            y_vars = [[] for _ in range(NUM_MODELS)]

            for i in range(NUM_MODELS):
                x[i] = x[i].to(device)
                if cfg["train_random_noise"] == "yes":
                    noise = torch.randn(x[i].shape).to(device)
                    adjusted_noise = noise * noise_variance
                    x[i] += adjusted_noise

                y_multi_preds = wrapped_prediction_models[i].predict_on_batch(x[i], iterations=cfg["num_trials"])
                y_pred_scores[i] = y_multi_preds.mean(dim=-1).reshape(-1)
                y_vars[i] = y_multi_preds.std(dim=-1).reshape(-1)

                y_preds[i] = (y_pred_scores[i] >= 0.5)
                all_pred_scores[i].extend(y_pred_scores[i].to('cpu').numpy())

            all_labels.extend(y.to('cpu').numpy())

            # Forward pass
            optimizer.zero_grad()
            final_predictions = fusion_model((x, y_pred_scores, y_vars))
            loss = criterion(final_predictions.reshape(-1), y)
            loss.backward()
            optimizer.step()

        #eval on dev set
        dev_metrics = evaluate_fusion_model(fusion_model, dev_loader, prediction_models, cfg)
        dev_loss = dev_metrics["loss"]
        dev_accuracy = dev_metrics["accuracy"]
        dev_balanced_accuracy = dev_metrics["weighted_accuracy"]
        dev_auroc = dev_metrics["auroc"]
        dev_f1 = dev_metrics["f1_score"]
        dev_ece = dev_metrics["ECE"]
        #print(f"Epoch {epoch}: dev accuracy: {dev_metrics['accuracy']}")

        if cfg['use_scheduler']=="yes":
            if cfg['scheduler']=='step':
                scheduler.step()
            else:
                scheduler.step(dev_loss)

        if dev_loss<best_dev_loss:
            best_model = copy.deepcopy(fusion_model)

            best_dev_loss = dev_loss
            best_dev_accuracy = dev_accuracy
            best_dev_balanced_accuracy = dev_balanced_accuracy
            best_dev_auroc = dev_auroc
            best_dev_f1 = dev_f1
            best_dev_ece = dev_ece

    test_metrics = evaluate_fusion_model(best_model, test_loader, prediction_models, cfg, split="test")
    print("\nDev Results\n" + "="*20)
    print({"dev_accuracy":best_dev_accuracy, "dev_balanced_accuracy":best_dev_balanced_accuracy, "dev_loss":best_dev_loss, "dev_auroc":best_dev_auroc, "dev_f1":best_dev_f1, "dev_ece":best_dev_ece})



    # Save best model
    torch.save(best_model.to('cpu').state_dict(),MODEL_PATH)
    if cfg["wandb"]=="True":
        wandb.save(MODEL_PATH)
        wandb.log(test_metrics)

    if cfg["fusion_model"] == "base":
        loaded_model = HybridFusionNetworkWithUncertainty(feature_shapes, cfg)
    else:
        loaded_model = HybridFusionNetworkWithUncertaintyBNN(feature_shapes, cfg)

    loaded_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    loaded_model = loaded_model.to(device)
    print("\nTest Results\n" + "="*20)
    print(evaluate_fusion_model(loaded_model, test_loader, prediction_models, cfg, split="test"))

if __name__ == "__main__":
    main()