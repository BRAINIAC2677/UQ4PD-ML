import optuna
from bnn.train_finger_tapping import main


def objective(trial):
    """Objective function for Optuna."""
    # Hyperparameter search space
    model = trial.suggest_categorical("model", ["bnn", "shallow_bnn"])
    seed = trial.suggest_int("seed", 0, 1000)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    max_epochs = trial.suggest_int("max_epochs", 5, 100)
    corr_thr = trial.suggest_float("corr_thr", 0.0, 1.0)
    scaler = trial.suggest_categorical("scaler", ["standard", "minmax"])
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adamw"])
    momentum = trial.suggest_float("momentum", 0.5, 0.99) if optimizer == "sgd" else None
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    beta1 = trial.suggest_float("beta1", 0.8, 0.99) if optimizer == "adamw" else None
    beta2 = trial.suggest_float("beta2", 0.9, 0.999) if optimizer == "adamw" else None
    kl_weight = trial.suggest_float("kl_weight", 1e-4, 1e-1, log=True)
    num_samples = trial.suggest_int("num_samples", 1, 10)

    args = {
        "model": model,
        "seed": seed,
        "lr": lr,
        "max_epochs": max_epochs,
        "corr_thr": corr_thr,
        "scaler": scaler,
        "optimizer": optimizer,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "kl_weight": kl_weight,
        "num_samples": num_samples,
    }

    # Call the main training function with the trial's hyperparameters
    results = main(args)

    # Return the metric to minimize (or maximize)
    return results[0]["test/cls/Acc"]  # Adjust based on your actual evaluation metric


def main_hpt():
    """Run Optuna hyperparameter tuning."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Adjust the number of trials as needed

    print("Best hyperparameters:", study.best_params)
    print("Best Accuracy:", study.best_value)


if __name__ == "__main__":
    main_hpt()
