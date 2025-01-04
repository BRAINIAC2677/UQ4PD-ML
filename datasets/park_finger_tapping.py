import re
import torch
import pandas as pd
from torch.utils.data import Dataset

import torch
import pandas as pd
from torch.utils.data import Dataset

def parse_patient_id(name):
    if name.startswith("NIH"):
        ID, *_ = name.split("-")
    elif name.endswith("finger_tapping.mp4"):
        *_, ID, _ = name.split("-")
    else:
        *_, ID, _, _, _ = name.split("_")
    return ID

def parse_date(name:str):
    match = re.search(r"\d{4}-\d{2}-\d{2}", name)
    date = match.group()
    return date

def concat_features(row):
    return list(row["features_right"]) + list(row["features_left"])

class ParkFingerTappingDataset(Dataset):
    def __init__(self, csv_path=None, ids=None, corr_thr=1, hand="both"):
        if csv_path is None:
            raise ValueError("csv_path must be provided")

        self.data = pd.read_csv(csv_path)
        self.features, self.labels, self.ids = self._extract_data(corr_thr, hand)

        # Filter based on IDs (if provided)
        if ids is not None:
            mask = self.ids.isin(ids)
            self.features = self.features[mask]
            self.labels = self.labels[mask]
            self.ids = self.ids[mask]


    def _extract_data(self, corr_thr, hand):
        data = self.data.copy()

        # Drop missing values
        data = data.dropna(subset = data.columns.difference(['Unnamed: 0','filename','Protocol','Participant_ID','Task',
                'Duration','FPS','Frame_Height','Frame_Width','gender','age','race',
                'ethnicity']), how='any')
 
        if hand != "both" and hand in ["left", "right"]:
            data = data[data["hand"] == hand]
            features = self._process_features(data, corr_thr)
            labels = 1.0 * (data["pd"] != "no").to_numpy()
            ids = data["filename"].apply(parse_patient_id)
            return features, labels, ids

        # Process for both hands
        right_data = data[data["hand"] == "right"]
        left_data = data[data["hand"] == "left"]

        features_right = self._process_features(right_data, corr_thr)
        labels_right = 1.0 * (right_data["pd"] != "no").to_numpy()
        ids_right = right_data["filename"].apply(parse_patient_id)
        dates_right = right_data["filename"].apply(parse_date)
        id_dates_right = ids_right + "#" + dates_right

        features_left = self._process_features(left_data, corr_thr)
        labels_left = 1.0 * (left_data["pd"] != "no").to_numpy()
        ids_left = left_data["filename"].apply(parse_patient_id)
        dates_left = left_data["filename"].apply(parse_date)
        id_dates_left = ids_left + "#" + dates_left

        # Merge left and right data
        df_right = pd.DataFrame({
            "features_right": list(features_right),
            "id_right": ids_right,
            "row_id": id_dates_right,
            "label_right": labels_right
        })

        df_left = pd.DataFrame({
            "features_left": list(features_left),
            "id_left": ids_left,
            "row_id": id_dates_left,
            "label_left": labels_left
        })

        df_both = pd.merge(df_right, df_left, how="inner", on="row_id")
        df_both = df_both.drop(columns=["label_left", "id_left"])
        df_both = df_both.rename(columns={"label_right": "label", "id_right": "id"})
        df_both["features"] = df_both.apply(concat_features, axis=1)

        features = df_both.loc[:, "features"]
        labels = df_both.loc[:, "label"]
        ids = df_both.loc[:, "id"]

        features = features.to_numpy()
        labels = labels.to_numpy()
        ids = df_both["id"]

        return features, labels, ids

    def _process_features(self, data, corr_thr):
        df_features = data.drop(columns=[
            "Unnamed: 0", "filename", "Protocol", "Participant_ID", "Task",
            "Duration", "FPS", "Frame_Height", "Frame_Width", "gender",
            "age", "race", "ethnicity", "pd", "hand"
        ])

        corr_matrix = df_features.corr()
        drop_cols = set()

        for i in range(len(corr_matrix.columns) - 1):
            for j in range(i + 1):
                if abs(corr_matrix.iloc[j, i + 1]) > corr_thr:
                    drop_cols.add(corr_matrix.columns[i + 1])

        df_features.drop(columns=drop_cols, inplace=True)

        features = df_features.to_numpy()
        return features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)


