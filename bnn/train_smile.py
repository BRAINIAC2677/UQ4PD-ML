from pathlib import Path
from torch import nn, optim

from torch_uncertainty.utils import TUTrainer
from torch_uncertainty.losses import ELBOLoss

from routines import ClassificationRoutine
from datamodules import ParkSmileDataModule
from models.park_smile import BNN, ShallowBNN


class ReshapeBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(ReshapeBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # Reshape targets to match model output shape
        targets = targets.view(-1, 1)
        return self.loss(outputs, targets)


def optim_bnn(model: nn.Module):
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    return optimizer


# datamodule
root = Path("./data/facial_expression_smile")
datamodule = ParkSmileDataModule(root=root, num_workers=7, test_ids_path=f'./data/test_set_participants.txt', dev_ids_path="./data/dev_set_participants.txt")

model = BNN(datamodule.num_features)
trainer = TUTrainer(accelerator="gpu", enable_progress_bar=True, log_every_n_steps=10, max_epochs=128)

loss = ELBOLoss(
    model=model,
    inner_loss = ReshapeBCEWithLogitsLoss(),
    kl_weight=1 / 1000,
    num_samples=5,
)

routine = ClassificationRoutine(
    model=model,
    num_classes=datamodule.num_classes,
    loss=loss,
    optim_recipe=optim_bnn(model),
    is_ensemble=True
)


trainer.fit(model=routine, datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)
