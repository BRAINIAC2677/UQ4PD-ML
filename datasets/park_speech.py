import torch
import pandas as pd
from torch.utils.data import Dataset

def parse_patient_id(name: str):
    if name.startswith("NIH"):
        [ID, *_] = name.split("-")
    elif name.endswith("-quick_brown_fox.mp4"):
        [*_, ID, _] = name.split("-")
    elif name.endswith("_quick_brown_fox.mp4"):
        [_, ID, _, _, _] = name.split("_")
    else:
        [*_, ID, _, _, _] = name.split("_")
    return ID

class ParkSpeechDataset(Dataset):
    def __init__(self, csv_path=None, ids=None, corr_thr=0.85):
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)

            self.data = self.data.dropna(subset = self.data.columns.difference(['Filename','Participant_ID', 'gender','age','race']), how='any')

            # Extract features
            self.features = self._extract_features(corr_thr)

            # Process labels
            self.labels = self.data['pd'].apply(lambda x: 0.0 if str(x) in ['no', '0.0'] else 1.0).to_numpy()

            # Parse patient IDs
            self.data['ID'] = self.data['Filename'].apply(parse_patient_id)
            self.ids = self.data['ID']

            # Filter based on IDs (if provided)
            if ids is not None:
                mask = self.data['ID'].isin(ids)
                self.features = self.features[mask]
                self.labels = self.labels[mask]
                self.ids = self.ids[mask]
          
        else:
            raise ValueError("csv_paths must be provided")

    def _extract_features(self, corr_thr):
        feature_columns = self.data.columns.difference(['Filename','Participant_ID', 'gender','age', 'race', 'pd'])

        features = self.data[feature_columns]

        corr_matrix = features.corr()
        drop_cols = set()

        for i in range(len(corr_matrix.columns) - 1):
            for j in range(i + 1):
                if abs(corr_matrix.iloc[j, i + 1]) > corr_thr:
                    drop_cols.add(corr_matrix.columns[i + 1])

        features = features.drop(columns=drop_cols)

        return features.to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)

