import torch
import argparse
import numpy as np
from pathlib import Path
from torch import nn, optim
from torch_uncertainty import TUTrainer

from losses import DECLoss
from datamodules import ParkFingerTappingDataModule
from routines import ClassificationRoutine
from models.park_finger_tapping import ANN, ShallowANN


def make_deterministic(seed: int):
    """Make experiments reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    """Main training function."""

    model = args['model']
    seed = args['seed']
    lr = args['lr']
    max_epochs = args['max_epochs']
    drop_prob = args['drop_prob']
    corr_thr = args['corr_thr']
    scaler = args['scaler']
    optimizer = args['optimizer']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    beta1 = args['beta1']
    beta2 = args['beta2']
    reg_weight = args['reg_weight']

    make_deterministic(seed)

    # Data preparation
    root = Path("./data/finger_tapping")
    datamodule = ParkFingerTappingDataModule(
        root=root,
        num_workers=7,
        scaler=scaler,
        corr_thr=corr_thr,
        test_ids_path="./data/test_set_participants.txt",
        dev_ids_path="./data/dev_set_participants.txt",
    )

    # Model definition
    if model == "ann":
        model = ANN(datamodule.num_features, drop_prob=drop_prob)
    elif model == "shallow_ann":
        model = ShallowANN(datamodule.num_features, drop_prob=drop_prob)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Optimizer
    if optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay= weight_decay,
        )
    elif optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr= lr,
            momentum= momentum,
            weight_decay= weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    loss = DECLoss(reg_weight=reg_weight)

    # Routine setup
    routine = ClassificationRoutine(
        num_classes=datamodule.num_classes,
        model=model,
        loss=loss,
        optim_recipe=optimizer,
    )

    # Trainer
    trainer = TUTrainer(
        accelerator="gpu", max_epochs= max_epochs, enable_progress_bar=False, log_every_n_steps=10
    )

    # Train and evaluate
    trainer.fit(model=routine, datamodule=datamodule)
    results = trainer.test(model=routine, datamodule=datamodule)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on Finger Tapping Data")
    parser.add_argument("--model", type=str, default="shallow_ann", choices=["ann", "shallow_ann"], help="Model type")
    parser.add_argument("--seed", type=int, default=604, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.0035999151237276687, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=85, help="Maximum epochs")
    parser.add_argument("--drop_prob", type=float, default=0.2685957816989365, help="Dropout probability")
    parser.add_argument("--corr_thr", type=float, default=0.8767490159878473, help="Correlation threshold for data")
    parser.add_argument("--scaler", type=str, default="minmax", choices=["standard", "minmax"], help="Scaler type")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"], help="Optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.07045409391333798, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.850924309225251, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.9966252622508455, help="Beta2 for AdamW")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="Regularization weight")

    args = parser.parse_args()
    args = vars(args)
    results = main(args)
    print(results)
