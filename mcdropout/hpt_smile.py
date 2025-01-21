import optuna
import argparse
from mcdropout.train_smile import main


def objective(trial):
    """Objective function for Optuna."""
    # Hyperparameter search space
    model = trial.suggest_categorical("model", ["ann", "shallow_ann"])
    seed = trial.suggest_int("seed", 0, 1000)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    max_epochs = trial.suggest_int("max_epochs", 5, 100)
    drop_prob = trial.suggest_float("drop_prob", 0.1, 0.5)
    num_estimators = trial.suggest_int("num_estimators", 100, 1000, step=100)
    corr_thr = trial.suggest_float("corr_thr", 0.0, 1.0)
    scaler = trial.suggest_categorical("scaler", ["standard", "minmax"])
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adamw"])
    momentum = trial.suggest_float("momentum", 0.5, 0.99) if optimizer == "sgd" else None
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    beta1 = trial.suggest_float("beta1", 0.8, 0.99) if optimizer == "adamw" else None
    beta2 = trial.suggest_float("beta2", 0.9, 0.999) if optimizer == "adamw" else None

    args = {
        "model": model,
        "seed": seed,
        "lr": lr,
        "max_epochs": max_epochs,
        "drop_prob": drop_prob,
        "num_estimators": num_estimators,
        "corr_thr": corr_thr,
        "scaler": scaler,
        "optimizer": optimizer,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
    }

    # Call the main training function with the trial's hyperparameters
    val_results, test_results = main(args)

    # Return the metric to minimize (or maximize)
    return val_results[0]["val/cls/AUROC"]  # Adjust based on your actual evaluation metric


def main_hpt(n_trials: int, log_file: str):
    """Run Optuna hyperparameter tuning."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)  # Adjust the number of trials as needed

    with open(log_file, "a") as f:
        f.write(f"Dataset: smile | Method: mcdropout\n")
        f.write(f"Best hyperparameters: {study.best_params}\n")
        f.write(f"Best value: {study.best_value}\n\n")

    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for MC-Dropout on SMILE dataset")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for Optuna") 
    parser.add_argument("--log_file", type=str, default="hpt_smile.log", help="Log file to save results")

    args = parser.parse_args()
    args = vars(args)
    main_hpt(**args)