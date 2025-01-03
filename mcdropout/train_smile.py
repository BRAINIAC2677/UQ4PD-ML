from pathlib import Path
from torch import nn, optim

from torch_uncertainty import TUTrainer

from mc_dropout import mc_dropout

from datamodules import ParkSmileDataModule
from routines import ClassificationRoutine
from models.park_smile import ANN, ShallowANN, BNNDrop


def optim_mcdropout(model: nn.Module):
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.03265227174722892,
    )
    return optimizer


root = Path("./data/facial_expression_smile")
trainer = TUTrainer(accelerator="cpu", max_epochs=64, enable_progress_bar=True, log_every_n_steps=10)
datamodule = ParkSmileDataModule(root=root, num_workers=7, test_ids_path=f'./data/test_set_participants.txt', dev_ids_path="./data/dev_set_participants.txt")
# model = ANN(datamodule.num_features, drop_prob=0.4)
model = BNNDrop(datamodule.num_features, drop_prob=0.4)
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