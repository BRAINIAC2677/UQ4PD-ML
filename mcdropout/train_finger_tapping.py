import torch 
import numpy as np
from pathlib import Path
from torch import nn, optim

from torch_uncertainty import TUTrainer

from mc_dropout import mc_dropout

from datamodules import ParkFingerTappingDataModule
from routines import ClassificationRoutine
from models.park_finger_tapping import ANN, ShallowANN


def make_deterministic(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def optim_mcdropout(model: nn.Module):
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.6246956232061768,
        momentum=0.8046223742478498
    )
    return optimizer


make_deterministic(790)
root = Path("./data/finger_tapping")
trainer = TUTrainer(accelerator="gpu", max_epochs=82, enable_progress_bar=True, log_every_n_steps=10)
datamodule = ParkFingerTappingDataModule(root=root, num_workers=7, scaler='standard', test_ids_path=f'./data/test_set_participants.txt', dev_ids_path="./data/dev_set_participants.txt")
model = ShallowANN(datamodule.num_features, drop_prob=0.24180259124462203)
mc_model = mc_dropout(model, num_estimators=1000, last_layer=False, on_batch=False)
routine = ClassificationRoutine(
    num_classes=datamodule.num_classes,
    model=mc_model,
    loss=nn.BCEWithLogitsLoss(),
    optim_recipe=optim_mcdropout(mc_model),
    is_ensemble=True,
)

trainer.fit(model=routine, datamodule=datamodule)
results = trainer.test(model=routine, datamodule=datamodule)