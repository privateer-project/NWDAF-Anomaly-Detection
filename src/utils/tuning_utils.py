from hyperopt import hp
from hyperopt.pyll.base import scope

# Define the search space for hyperparameters
search_space = {
    # Model hyperparameters
    "hidden_dim1": scope.int(hp.quniform("hidden_dim1", 30, 100, 10)),
    "hidden_dim2": scope.int(hp.quniform("hidden_dim2", 100, 200, 10)),
    "dropout": hp.uniform("dropout", 0.0, 0.5),
    "layer_norm_flag": hp.choice("layer_norm_flag", [True, False]),
    "num_layers": scope.int(hp.quniform("num_layers", 1, 2, 1)),
    # Training-related hyperparameters
    # "lr": hp.loguniform("lr", -5, -3),  # Learning rate between 0.00001 and 0.001
    "batch_size": hp.choice("batch_size", [16, 32, 64]),
    "num_epochs": scope.int(hp.quniform("num_epochs", 32, 128, 16)),
    "time_window_length": scope.int(hp.quniform("time_window_length", 60, 160, 10)),
    # "step_size": scope.int(hp.quniform("step_size", 30, 60, 10)),
    # Optimizer type (choices between Adam and AdamW)
    # "optimizer_type": hp.choice("optimizer_type", ["Adam", "AdamW"]),
    # Criterion type (choices between MSE and L1 Loss)
    # "criterion": hp.choice("criterion", ["MSE", "L1"]),
}
