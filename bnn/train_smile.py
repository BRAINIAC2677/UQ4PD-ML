import torch
import argparse
import numpy as np
from pathlib import Path
from torch import nn, optim
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import ELBOLoss

from datamodules import ParkSmileDataModule
from routines.uspark import ClassificationRoutine
from models.park_finger_tapping import BNN, ShallowBNN


def make_deterministic(seed: int):
    """Make experiments reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ReshapeBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(ReshapeBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # Reshape targets to match model output shape
        targets = targets.view(-1, 1)
        return self.loss(outputs, targets)


def main(args):
    """Main training function."""

    model = args['model']
    seed = args['seed']
    lr = args['lr']
    max_epochs = args['max_epochs']
    corr_thr = args['corr_thr']
    scaler = args['scaler']
    optimizer = args['optimizer']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    beta1 = args['beta1']
    beta2 = args['beta2']
    kl_weight = args['kl_weight']
    num_samples = args['num_samples']

    make_deterministic(seed)

    # Data preparation
    root = Path("./data/uspark/facial_expression_smile")
    datamodule = ParkSmileDataModule(
        root=root,
        num_workers=7,
        scaler=scaler,
        corr_thr=corr_thr,
        test_ids_path="./data/uspark/test_set_participants.txt",
        dev_ids_path="./data/uspark/dev_set_participants.txt",
    )

    # Model definition
    if model == "bnn":
        model = BNN(datamodule.num_features)
    elif model == "shallow_bnn":
        model = ShallowBNN(datamodule.num_features)
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

    loss = ELBOLoss(
        model=model,
        inner_loss = ReshapeBCEWithLogitsLoss(),
        kl_weight=kl_weight,
        num_samples=num_samples,
    )

    # Routine setup
    routine = ClassificationRoutine(
        num_classes=datamodule.num_classes,
        model=model,
        loss=loss,
        optim_recipe=optimizer,
        is_ensemble=True,
    )

    # Trainer
    trainer = TUTrainer(
        accelerator="gpu", max_epochs= max_epochs, enable_progress_bar=False, log_every_n_steps=10
    )

    # Train and evaluate
    trainer.fit(model=routine, datamodule=datamodule)
    test_results = trainer.test(model=routine, datamodule=datamodule)
    val_results = trainer.validate(model=routine, datamodule=datamodule)

    return (val_results, test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a bnn model smile data")
    parser.add_argument("--model", type=str, default="shallow_bnn", choices=["bnn", "shallow_bnn"], help="Model type")
    parser.add_argument("--seed", type=int, default=351, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.0032308494043844956, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=22, help="Maximum epochs")
    parser.add_argument("--corr_thr", type=float, default=0.7170944479791501, help="Correlation threshold for data")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"], help="Scaler type")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"], help="Optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0004467915509615761, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9198081403627488, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.9243536203069861, help="Beta2 for AdamW")
    parser.add_argument("--kl_weight", type=float, default=0.009123695409647019, help="KL weight for ELBO loss")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples for ELBO loss")

    args = parser.parse_args()
    args = vars(args)
    val_results, test_results = main(args)
    