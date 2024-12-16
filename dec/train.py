from pathlib import Path
from torch import nn, optim

from torch_uncertainty.utils import TUTrainer
from losses import DECLoss

from routines import ClassificationRoutine
from datamodules import ParkSmileDataModule
from models.park_smile import BNN, ShallowBNN, ANN, ShallowANN


def optim_dec(model: nn.Module) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return {"optimizer": optimizer, "lr_scheduler": exp_lr_scheduler}

# datamodule
root = Path("./data/facial_expression_smile")
datamodule = ParkSmileDataModule(root=root, num_workers=7, test_ids_path=f'./data/test_set_participants.txt', dev_ids_path="./data/dev_set_participants.txt")

print(f'size of train set: {len(datamodule.train_dataloader().dataset)}')
print(f'size of val set: {len(datamodule.val_dataloader().dataset)}')
print(f'size of test set: {len(datamodule.test_dataloader().dataset)}')


model = ShallowANN(datamodule.num_features, drop_prob=0.4)
trainer = TUTrainer(accelerator="cpu", enable_progress_bar=False, log_every_n_steps=10, max_epochs=2)

loss = DECLoss(reg_weight=1e-2)

routine = ClassificationRoutine(
    model=model,
    num_classes=datamodule.num_classes,
    loss=loss,
    optim_recipe=optim_dec(model),
)

trainer.fit(model=routine, datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)
