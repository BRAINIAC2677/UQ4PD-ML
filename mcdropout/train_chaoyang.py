import torch
import argparse
import numpy as np
from pathlib import Path
from torch import nn, optim
from torch_uncertainty import TUTrainer
from torch_uncertainty.models.resnet import resnet

from mc_dropout import mc_dropout
from datamodules import ChaoyangDataModule
from routines import ClassificationRoutine


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
    num_estimators = args['num_estimators']
    optimizer = args['optimizer']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    beta1 = args['beta1']
    beta2 = args['beta2']

    make_deterministic(seed)

    # Data preparation
    root = Path("./data/chaoyang")
    datamodule = ChaoyangDataModule(
        root=root,
        num_workers=7,
    )

    print(f"len(train): {len(datamodule.train)}")
    print(f"len(test): {len(datamodule.test)}")

    # Model definition
    if model == "resnet":
        model = resnet(arch=34, in_channels=datamodule.num_channels, num_classes=datamodule.num_classes, dropout_rate=drop_prob)
    else:
        raise ValueError(f"Unknown model: {model}")

    mc_model = mc_dropout(model, num_estimators=num_estimators, last_layer=False, on_batch=False)

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

    # Routine setup
    routine = ClassificationRoutine(
        num_classes=datamodule.num_classes,
        model=mc_model,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=optimizer,
        is_ensemble=True,
    )

    # Trainer
    trainer = TUTrainer(
        accelerator="gpu", max_epochs= max_epochs, enable_progress_bar=True, log_every_n_steps=10
    )

    # Train and evaluate
    trainer.fit(model=routine, datamodule=datamodule)
    results = trainer.test(model=routine, datamodule=datamodule)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a mcdropout model on smile data")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet"], help="Model type")
    parser.add_argument("--seed", type=int, default=914, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.005636638313326733, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum epochs")
    parser.add_argument("--drop_prob", type=float, default=0.23801571998298293, help="Dropout probability")
    parser.add_argument("--num_estimators", type=int, default=700, help="Number of estimators for MC Dropout")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"], help="Optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD (not used in adamw)")
    parser.add_argument("--weight_decay", type=float, default=0.0631540840367034, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.843677246295737, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.9202703944120154, help="Beta2 for AdamW")

    args = parser.parse_args()
    args = vars(args)

    results = main(args)
    print(results)


